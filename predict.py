import cv2 as cv
import numpy as np
import torch
import utils.data_loader as dt_loader

from nets.yoloV3 import YoloV3



# read classes
classes_path    = 'model_data/voc_classes.txt'
class_names, num_classes = dt_loader.get_classes(classes_path)

# model data
input_shape     = [416, 416]
yolo = YoloV3(num_classes, input_shape)
yolo = yolo.cuda()

model_path = "model_data/trained_models/ep001-loss28.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo.load_state_dict(torch.load(model_path, map_location=device))


# read a image and preprocess
img_path = "data/VOC2007/JPEGImages/000001.jpg"
im_src = cv.imread(img_path, cv.IMREAD_COLOR)
im_rgb = cv.cvtColor(im_src, cv.COLOR_BGR2RGB)

in_H = input_shape[0]
in_W = input_shape[1]
im_H, im_W, chs = im_rgb.shape
# scale image with fixed ratio
scalar = min(in_H/im_H, in_W/im_W)
new_h = int(im_H*scalar)
new_w = int(im_W*scalar)
dx = (in_W - new_w)//2
dy = (in_H - new_h)//2

im_scaled = cv.resize(im_rgb, dsize=(new_w, new_h))
im_input = np.ones((in_H, in_W, chs))*128

# copy scaled img to input img
im_input[dy:(dy+new_h), dx:(dx+new_w)] = np.array(im_scaled)
im_input = np.expand_dims(np.float32(im_input),axis=0)

tensor_NCHW = torch.tensor(dt_loader.conversion_NCWH(im_input)).cuda()
outputs = yolo.forward(tensor_NCHW)

boxes_withInfo = yolo.decodeBoxWithInfo(outputs)

results = yolo.nonMaxSuppression(boxes_withInfo)
i=0
print("objects num: %d" % len(results[i]))

# convert ratio of x,y,w,h to the scale of source image
input_shape = np.array(input_shape)
image_shape = np.array([im_H, im_W])
new_shape = np.round(image_shape * np.min(input_shape/image_shape))
offset  = (input_shape - new_shape)/2./input_shape
scale   = input_shape/new_shape


disp_img = im_src
for bbox in results[i]:
    
    box_xy = bbox[0:2]
    box_wh = bbox[2:4]
    
    box_xy  = (box_xy - offset[::-1]) * scale[::-1]
    box_wh *= scale
    
    box_xy *= image_shape
    box_wh *= image_shape
    
    tpL     = box_xy - (box_wh / 2.)
    btR     = box_xy + (box_wh / 2.)
    
    tpL = np.int16(tpL)
    btR = np.int16(btR)
    
    cv.rectangle(disp_img,tpL,btR,(0,255,0),2)
    
cv.imshow("yolo detect",disp_img)
cv.waitKey(0)



