import time
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict

import nets.darknet as dnet

# init weights of networks using different types
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

class YoloV3(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        # init darknet53
        self.backbone = dnet.Darknet53()
        
        # num of classses
        self.numClasses = num_classes
        self.input_shape = input_shape
        
        # ignore bbox
        self.NoobjIOU_th = 0.5
        
        # NMS threshold
        self.nms_thres = 0.4
        
        # confidence threshold
        self.conf_thres = 0.5
        
        # define anchors : [Width, Height]
        self.anchors = [
                        [[116,90], [156,198], [373,326]],   # 13x13
                        [[30,61], [62,45], [59,119]],       # 26x26
                        [[10,13], [16,30], [33,23]]         # 52x52
                        ]
        self.anchors_ids = [[6,7,8], [3,4,5], [0,1,2]]
        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters
        
        # 13x13 path
        out0_chs = (num_classes+1+4)*len(self.anchors[0])
        
        # final output 13x13
        # [13,13,1024] -> [13,13,512]
        self.body_layer0 = self.DBL_5layers(out_filters[-1] + 0, [out_filters[-2], out_filters[-1]])
        # [13,13,512] -> [13,13,1024]
        self.body_layer0_final_DBL = self.DBL_block(out_filters[-2], out_filters[-1], 3)
        # 1x1 conv:reduce chs to output and keep scale
        # [13,13,1024] -> [13,13,(20+1+4)*3]
        self.body_layer0_final_conv = nn.Conv2d(in_channels=out_filters[-1], out_channels=out0_chs, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 26x26 path
        out1_chs = (num_classes+1+4)*len(self.anchors[1])
        
        # concating
        # [13,13,512] -> [13,13,256]
        self.body_layer1_dbl = self.DBL_block(out_filters[-2], out_filters[-3], 1)
        # [13,13,256] -> [26,26,256]
        self.body_layer1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # final output 26x26
        # [26,26,512+256] -> [26,26,256]
        self.body_layer1 = self.DBL_5layers(out_filters[-2] + out_filters[-3], [out_filters[-3], out_filters[-2]])
        # [26,26,256] -> [26,26,512]
        self.body_layer1_final_DBL = self.DBL_block(out_filters[-3], out_filters[-2], 3)
        # 1x1 conv:reduce chs to output and keep scale
        # [26,26,512] -> [26,26,(20+1+4)*3]
        self.body_layer1_final_conv = nn.Conv2d(in_channels=out_filters[-2], out_channels=out1_chs, kernel_size=1, stride=1, padding=0, bias=True)
        
        
        # 52x52 path
        out2_chs = (num_classes+1+4)*len(self.anchors[2])
        
        # concating
        # [26,26,256] -> [26,26,128]
        self.body_layer2_dbl = self.DBL_block(out_filters[-3], out_filters[-4], 1)
        # [26,26,256] -> [52,52,128]
        self.body_layer2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # final output 26x26
        # [52,52,256+128] -> [52,52,128]
        self.body_layer2 = self.DBL_5layers(out_filters[-3]+128, [out_filters[-4], out_filters[-3]])
        # [52,52,128] -> [52,52,256]
        self.body_layer2_final_DBL = self.DBL_block(out_filters[-4], out_filters[-3], 3)
        # 1x1 conv:increase chs to output and keep scale
        # [52,52,256] -> [52,52,(20+1+4)*3]
        self.body_layer2_final_conv = nn.Conv2d(in_channels=out_filters[-3], out_channels=out2_chs, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        
        # get last three feats in different scales
        t_0 = time.time()
        feats2,feats1,feats0 = self.backbone(x)
        t_1 = time.time()
        ############ 13x13 path
        x0 = self.body_layer0(feats0)
        # output 0
        out0 = self.body_layer0_final_DBL(x0)
        out0 = self.body_layer0_final_conv(out0)
        t_2 = time.time()
        ############ 26x26 path
        # concat x0 and feat1
        x0_cat = self.body_layer1_dbl(x0)
        x0_cat = self.body_layer1_upsample(x0_cat)
        x1 =  torch.cat([x0_cat, feats1],1)
        x1 = self.body_layer1(x1)
        # output 1
        out1 = self.body_layer1_final_DBL(x1)
        out1 = self.body_layer1_final_conv(out1)
        t_3 = time.time()
        ############ 52x52 path
        # concat x1 and feat2
        x1_cat = self.body_layer2_dbl(x1)
        x1_cat = self.body_layer2_upsample(x1_cat)
        x2 = torch.cat([x1_cat, feats2],1)
        x2 = self.body_layer2(x2)
        # output 2
        out2 = self.body_layer2_final_DBL(x2)
        out2 = self.body_layer2_final_conv(out2)
        t_4 =time.time()
        
        # print("backbone time: %.3f, 13x13 time:%.3f, 26x26 time:%.3f, 52x52 time: %.3f"%(t_1-t_0,t_2-t_1,t_3-t_2,t_4-t_3))
        
        return out0,out1,out2
        
        
        
    # DBL: conv+BN+leakyRelu
    def DBL_block(self, in_filter, out_filter, kernel_size):
        
        pad = (kernel_size - 1)//2 if kernel_size else 0
        DBL = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=in_filter, out_channels=out_filter, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(out_filter)),
            ("relu", nn.ReLU(0.1))
        ]))
    
        return DBL
    
    # 5xDBL
    def DBL_5layers(self, in_filter, filters):
        layers = nn.Sequential(
            self.DBL_block(in_filter,  filters[0], 1), # 1x1 conv
            self.DBL_block(filters[0], filters[1], 3),
            self.DBL_block(filters[1], filters[0], 1), # 1x1 conv
            self.DBL_block(filters[0], filters[1], 3),
            self.DBL_block(filters[1], filters[0], 1), # 1x1 conv
        )
        
        return layers
    
    # calculate loss for each head
    def loss(self, heads, gt):
        loss = 0.0
        num_pos_objs = 0
        for scaleIdx, head in enumerate(heads):
            
            
            batch_size = head.size(0)
            anchors = self.anchors[scaleIdx]
            anchors_size = len(anchors)
            
            # bs, 3*(5+num_classes), 13, 13
            # bs, 3*(5+num_classes), 26, 26
            # bs, 3*(5+num_classes), 52, 52
            
            feats_H = head.size(2)
            feats_W = head.size(3)
            
            stride_H = self.input_shape[0] // feats_H
            stride_W = self.input_shape[1] // feats_W
            
            scaled_anchors = [(a_w / stride_W, a_h /stride_H) for a_w, a_h in anchors]
            
            ####### Predictions
            # head [b,c,h,w] -> [b,anchors,rect+class,h,w] -> [b,anchors,h,w,rect+class]
            preds = head.view(batch_size, anchors_size,5+self.numClasses, feats_H, feats_W).permute(0,1,3,4,2).contiguous()
            
            # prediction rect
            pred_x = torch.sigmoid(preds[...,0])
            pred_y = torch.sigmoid(preds[...,1])
            pred_h = preds[...,2]
            pred_w = preds[...,3]
            
            # confidence
            pred_conf = torch.sigmoid(preds[...,4])
            
            # classification
            pred_cls = torch.sigmoid(preds[...,5:])
            
            ####### Groud Truth
            # [b,3,H,W]
            noobj_mask = torch.ones(batch_size, anchors_size, feats_H, feats_W, requires_grad = False)
            # [b,3,H,W]
            box_loss_scale = torch.zeros(batch_size, anchors_size, feats_H, feats_W, requires_grad = False)
            # [b,3,H,W,5+C]
            label_true = torch.zeros(batch_size, anchors_size, feats_H, feats_W, 5+self.numClasses, requires_grad=False)
            
            # iterate each batch and fill value above 3 
            for b in range(batch_size):
                
                # calc position in feats
                one_batch = torch.zeros_like(gt[b])
                one_batch[:,[0,2]] = gt[b][:,[0,2]] * feats_W
                one_batch[:,[1,3]] = gt[b][:,[1,3]] * feats_H
                one_batch[:,4] = gt[b][:,4]
                
                # [num_box, 4]
                gt_box = torch.cuda.FloatTensor(torch.cat((torch.zeros((one_batch.size(0),2)).cuda(), one_batch[:,2:4]), 1))
                
                # [9, 4]
                numOfanchors = len(self.anchors)*len(self.anchors[0])
                anchors_shapes = torch.cuda.FloatTensor(torch.cat((torch.zeros((numOfanchors, 2)).cuda(), torch.cuda.FloatTensor(self.anchors).reshape(-1,2)), 1))
                
                # calc IOU between 1 truth box and 9 prior box: don't need x,y. find the best anchor wrt groud truth box 
                gt_anchors_IOUs = self.calculateIOU(gt_box, anchors_shapes)
                # find best IOU from 9 priors WRT truth box
                best_priors = torch.argmax(gt_anchors_IOUs, dim=-1)
                
                # iterate all truth box
                for n, best_pId in enumerate(best_priors):
                    
                    if best_pId not in self.anchors_ids[scaleIdx]: continue
                    
                    # prior id from a scale, one of the three priors
                    k = self.anchors_ids[scaleIdx].index(best_pId)
                    
                    # locate the grid i, j
                    c_i = torch.floor(one_batch[n, 0]).long()
                    c_j = torch.floor(one_batch[n, 1]).long()
                    
                    # truth classes 
                    c = one_batch[n, 4].long()
                    # mark noobj as 0: (grid i,j) has object
                    noobj_mask[b,k,c_j, c_i] = 0
                    
                    # tx, ty, tw, th
                    label_true[b,k,c_j,c_i,0] = one_batch[n,0] - c_i.float()
                    label_true[b,k,c_j,c_i,1] = one_batch[n,1] - c_j.float()
                    
                    prior_w = scaled_anchors[k][0]
                    prior_h = scaled_anchors[k][1]
                    label_true[b,k,c_j,c_i,2] = torch.log(one_batch[n,2]/ prior_w)
                    label_true[b,k,c_j,c_i,3] = torch.log(one_batch[n,3]/ prior_h)
                    
                    # assgin confidence and classes
                    label_true[b,k,c_j,c_i,4] = 1
                    label_true[b,k,c_j,c_i,5+c] = 1.0 # c is the index of classes as same as class label 
                    
                    # consider box W,H in large and small case
                    box_loss_scale[b,k,c_j,c_i] = one_batch[n,2]*one_batch[n,3]/feats_W/feats_H
                    
            ####### Compute Loss
            # define if a grid has object
            noobj_mask = self.genNoObjMask(pred_x,pred_y,pred_h,pred_w, gt, scaled_anchors, feats_H, feats_W, noobj_mask)
            
            if self.cuda:
                label_true = label_true.cuda()
                noobj_mask = noobj_mask.cuda()
                box_loss_scale = box_loss_scale.cuda()
            
            box_loss_scale = 2 - box_loss_scale
            
            #  center x,y loss
            loss_x = torch.sum(self.BCELoss(pred_x, label_true[...,0])*box_loss_scale*label_true[...,4])
            loss_y = torch.sum(self.BCELoss(pred_y, label_true[...,1])*box_loss_scale*label_true[...,4])
            #  w,h loss
            loss_w = torch.sum(self.MSELoss(pred_w, label_true[...,2])*0.5*box_loss_scale*label_true[...,4])
            loss_h = torch.sum(self.MSELoss(pred_h, label_true[...,3])*0.5*box_loss_scale*label_true[...,4])
            #  confidence loss
            loss_conf = torch.sum(self.BCELoss(pred_conf, label_true[...,4])*label_true[...,4]) + \
                        torch.sum(self.BCELoss(pred_conf, label_true[...,4])*noobj_mask)
            #  classes loss
            loss_cls = torch.sum(self.BCELoss(pred_cls[label_true[...,4]==1], label_true[...,5:][label_true[...,4]==1]))
            
            curr_loss =  loss_x+loss_y+loss_w+loss_h+loss_conf+loss_cls
            num_pos = torch.sum(label_true[...,4]) # num of positive object in truth
            num_pos = torch.max(num_pos, torch.ones_like(num_pos)) # for one batch, at leaest has one postive object
            
            loss = loss+curr_loss
            num_pos_objs = num_pos_objs+num_pos
            
       
        return loss/num_pos_objs
    
    # Mean square error: define loss of box location and size
    def MSELoss(self, pred, label):
        return torch.pow(pred-label, 2)
    
    # binary cross entropy: define loss of confidence and classification
    def BCELoss(self, pred, label):
        pred = torch.clip(pred, 1e-7, 1.0-1e-7) # max pred is close to 1.0, min pred is close to 0.0
        return -label*torch.log(pred)-(1.0-label)*torch.log(1.0-pred)
    
    def calculateIOU(self, box1, box2):
        
        # box1: top-left point and bot-right point
        b1_x1 = box1[:,0] - box1[:,2]/2 # top-left X: centerX - W/2
        b1_y1 = box1[:,1] - box1[:,3]/2 # top-left Y: centerY - H/2
        
        b1_x2 = box1[:,0] + box1[:,2]/2 # bot-rightX: centerX + W/2
        b1_y2 = box1[:,1] + box1[:,3]/2 # bot_rightY: centerY + H/2
        
        # box2: top-left point and bot-right point
        b2_x1 = box2[:,0] - box2[:,2]/2 # top-left X: centerX - W/2
        b2_y1 = box2[:,1] - box2[:,3]/2 # top-left Y: centerY - H/2
        
        b2_x2 = box2[:,0] + box2[:,2]/2 # bot-rightX: centerX + W/2
        b2_y2 = box2[:,1] + box2[:,3]/2 # bot_rightY: centerY + H/2
        
        # assgin box1, box2 to the format of diagonal pts
        box1_dp = torch.zeros_like(box1)
        box1_dp[:,0] = b1_x1
        box1_dp[:,1] = b1_y1
        box1_dp[:,2] = b1_x2
        box1_dp[:,3] = b1_y2
        
        box2_dp = torch.zeros_like(box2)
        box2_dp[:,0] = b2_x1
        box2_dp[:,1] = b2_y1
        box2_dp[:,2] = b2_x2
        box2_dp[:,3] = b2_y2
        
        num1 = box1.size(0)
        num2 = box2.size(0)
        
        # calc intersection area
        # top,left are defined in image coordinates
        bot_right_xy  = torch.min(box1_dp[:, 2:].unsqueeze(1).expand(num1, num2, 2), box2_dp[:, 2:].unsqueeze(0).expand(num1, num2, 2))
        top_left_xy   = torch.max(box1_dp[:, :2].unsqueeze(1).expand(num1, num2, 2), box2_dp[:, :2].unsqueeze(0).expand(num1, num2, 2))
        intersection  = torch.clamp((bot_right_xy - top_left_xy), min=0)
        intersect_area  = intersection[:, :, 0] * intersection[:, :, 1]
        
        box1_area = ((box1_dp[:,2] - box1_dp[:,0])*(box1_dp[:,3] - box1_dp[:,1])).unsqueeze(1).expand_as(intersect_area)
        box2_area = ((box2_dp[:,2] - box2_dp[:,0])*(box2_dp[:,3] - box2_dp[:,1])).unsqueeze(0).expand_as(intersect_area)

        
        # IOU
        union = (box1_area+box2_area) - intersect_area
        iou = intersect_area / union
        return iou
    
    # assign mask with object when the preds box is highly overlay with truth box
    # noobj: 1 is no obj 
    def genNoObjMask(self,x,y,h,w,gts,scaled_anchor, feat_H, feat_W, noobj_mask):
        
        batchSize = len(gts)
        
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        
        # generate grids on featmap
        grid_x = torch.linspace(0, feat_W-1, feat_W).repeat(feat_H, 1)
        grid_x = grid_x.repeat(int(batchSize*3),1,1)
        grid_x = grid_x.view(x.shape).type(FloatTensor)
        
        grid_y = torch.linspace(0, feat_H-1, feat_H).repeat(feat_W, 1).t()
        grid_y = grid_y.repeat(int(batchSize*3),1,1)
        grid_y = grid_y.view(y.shape).type(FloatTensor)
        
        # prior box W,H based on anchors
        anchor_w = FloatTensor(scaled_anchor).index_select(1,LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchor).index_select(1,LongTensor([1]))
        
        anchor_w = anchor_w.repeat(batchSize, 1).repeat(1,1,feat_H*feat_W).view(w.shape)
        anchor_h = anchor_h.repeat(batchSize, 1).repeat(1,1,feat_H,feat_W).view(h.shape)
        
        # predictions box
        pred_box_x = torch.unsqueeze(x.data + grid_x, -1)
        pred_box_y = torch.unsqueeze(y.data + grid_y, -1)
        pred_box_w = torch.unsqueeze(torch.exp(w.data)*anchor_w, -1)
        pred_box_h = torch.unsqueeze(torch.exp(h.data)*anchor_h, -1)
        pred_boxes = torch.cat([pred_box_x,pred_box_y,pred_box_w,pred_box_h], dim=-1)
        
        
        for b in range(batchSize):
            
            curr_pred_box = pred_boxes[b].view(-1,4)
            
            if len(gts[b])>0:
                
                one_batch = torch.zeros_like(gts[b])
                
                # generate truth box: one image could have multiple truth boxes
                one_batch[:,[0,2]] = gts[b][:,[0,2]]* feat_W
                one_batch[:,[1,3]] = gts[b][:,[1,3]]* feat_H
                one_batch = one_batch[:,:4]
                
                # compute IOU, compare preds box to each truth box
                ious = self.calculateIOU(one_batch, curr_pred_box)
                
                # find the max iou between a pred box and truth box.
                iou_max, _ = torch.max(ious, dim=0)
                iou_max = iou_max.view(pred_boxes[b].size()[:3])
                
                noobj_mask[b][iou_max>self.NoobjIOU_th] = 0
        
        return noobj_mask
        
    # decode bboxes attached with confi and classes
    def decodeBoxWithInfo(self, heads):
        outputs = []
        for i, input in enumerate(heads):
            #-----------------------------------------------#
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            scaled_anchors =[]
            anchors = self.anchors[i]
            for anchor in anchors:
                scaled_anchors.append([anchor[0]/stride_w, anchor[1]/stride_h])
                

            #-----------------------------------------------#
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            #-----------------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_ids[i]), 5+self.numClasses, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()


            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            conf        = torch.sigmoid(prediction[..., 4])
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_ids[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_ids[i]), 1, 1).view(y.shape).type(FloatTensor)

            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)


            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h

            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.numClasses)), -1)
            outputs.append(output.data)
        return outputs
    
    # Nonmaximum suppression
    def nonMaxSuppression(self, decodedBoxes):
        decodedBoxes = torch.cat(decodedBoxes, 1)
        
        # convert x,y,h,w -> topleft, botright
        box_corner = decodedBoxes.new(decodedBoxes.shape)
        box_corner[:, :, 0] = decodedBoxes[:, :, 0] - decodedBoxes[:, :, 2] / 2
        box_corner[:, :, 1] = decodedBoxes[:, :, 1] - decodedBoxes[:, :, 3] / 2
        box_corner[:, :, 2] = decodedBoxes[:, :, 0] + decodedBoxes[:, :, 2] / 2
        box_corner[:, :, 3] = decodedBoxes[:, :, 1] + decodedBoxes[:, :, 3] / 2
        decodedBoxes[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(decodedBoxes))]
        
        for i, image_pred in enumerate(decodedBoxes):
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + self.numClasses], 1, keepdim=True)
            
            # keep boxes that Pr(class|object) is greater thant conf_thres
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= self.conf_thres).squeeze()
            
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            unique_labels = detections[:, -1].cpu().unique()
            if decodedBoxes.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()
             
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]

                # sorting object by conf*class
                _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                detections_class = detections_class[conf_sort_index]
                
                # suppress non maximum box
                max_detections = []
                while detections_class.size(0):
                    # iterate the rest of the class, and check iou with nms_thres
                    max_detections.append(detections_class[0].unsqueeze(0))
                    if len(detections_class) == 1:
                        break
                    ious = self.calculateIOU(max_detections[-1], detections_class[1:])
                    detections_class = detections_class[1:][ious < self.nms_thres]
                    
                max_detections = torch.cat(max_detections).data
                
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            
                
                