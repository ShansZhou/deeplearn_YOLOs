import torch.nn as nn


import nets.darknet as dnet


class YoloV3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.darkNet = dnet.Darknet53()
        
        
        
        