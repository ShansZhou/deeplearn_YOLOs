import torch.nn as nn
import torch
from collections import OrderedDict

import nets.darknet as dnet


class YoloV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # init darknet53
        self.backbone = dnet.Darknet53()
        
        # define anchors
        self.anchors = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        
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
        feats2,feats1,feats0 = self.backbone(x)
        
        ############ 13x13 path
        x0 = self.body_layer0(feats0)
        # output 0
        out0 = self.body_layer0_final_DBL(x0)
        out0 = self.body_layer0_final_conv(out0)
        
        ############ 13x13 path
        # concat x0 and feat1
        x0_cat = self.body_layer1_dbl(x0)
        x0_cat = self.body_layer1_upsample(x0_cat)
        x1 =  torch.cat([x0_cat, feats1],1)
        x1 = self.body_layer1(x1)
        # output 1
        out1 = self.body_layer1_final_DBL(x1)
        out1 = self.body_layer1_final_conv(out1)
        
        ############ 13x13 path
        # concat x1 and feat2
        x1_cat = self.body_layer2_dbl(x1)
        x1_cat = self.body_layer2_upsample(x1_cat)
        x2 = torch.cat([x1_cat, feats2],1)
        x2 = self.body_layer2(x2)
        # output 2
        out2 = self.body_layer2_final_DBL(x2)
        out2 = self.body_layer2_final_conv(out2)
        
        
        ################ decoding heads
        
        
        
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
    
        