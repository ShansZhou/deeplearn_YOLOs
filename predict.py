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

model_path = "model_data/ep001-loss28.917.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo.load_state_dict(torch.load(model_path, map_location=device))


# read a image and preprocess
img_path = "data/VOC2007/JPEGImages/000001.jpg"
image_shape = cv.imread(img_path, cv.IMREAD_COLOR)
im_rgb = cv.cvtColor(image_shape, cv.COLOR_BGR2RGB)

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




