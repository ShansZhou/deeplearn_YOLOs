import torch.nn as nn


import nets.darknet as dnet


class YoloV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # init darknet53
        self.backbone = dnet.Darknet53()
        
        # get three feats for heads
        out_feats = self.backbone.layers_out_filters
        
        
    def forward(self, x):
        
        feats2,feats1,feats0 = self.backbone(x)
        
        
        
        
        