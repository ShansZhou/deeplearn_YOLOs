import torch.nn as nn
import numpy as np

from collections import OrderedDict

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        
        self.layers = [1,2,8,8,4] # residual blocks in 5 layers
        self.in_filters = 32
        
        # [416,416,3] -> [416,416,32]
        self.conv1 = nn.Conv2d(3, self.in_filters, kernel_size=3, stride=1, padding=1, bias= False)
        self.bn1 = nn.BatchNorm2d(self.in_filters)
        self.relu1 = nn.ReLU(0.1)
        
        # make layers 1 -> 2 -> 8 -> 8 -> 4
        # [416,416,32] -> [208,208,64]
        self.layer1 = self.DBL_layer([32, 64], self.layers[0])
        
        # [208,208,64] -> [104,104,128]
        self.layer2 = self.DBL_layer([64, 128], self.layers[1])
        
        # [104,104,128] -> [52,52,256]
        self.layer3 = self.DBL_layer([128, 256], self.layers[2])
        
        # [52,52,256] -> [26,26,512]
        self.layer4 = self.DBL_layer([256, 512], self.layers[3])
        
        # [26,26,512] -> [14,14,1024]
        self.layer5 = self.DBL_layer([512, 1024], self.layers[4])
        

        self.layers_out_filters = [64, 128, 256, 512, 1024]
        
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    # ResN block
    def DBL_layer(self, filters, residual_blocks):
        layers = []
        
        # downsampling block
        ds_conv = ("ds_conv",
                    nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=2, padding=1, bias=False))
        ds_bn   = ("ds_bn",
                    nn.BatchNorm2d(filters[1]))
        ds_relu = ("ds_relu",
                    nn.LeakyReLU(0.1))
        
        layers.append(ds_conv)
        layers.append(ds_bn)
        layers.append(ds_relu)
        
        # append residual N blocks
        for i in range(0, residual_blocks):
            res_block = ("residual_{}".format(i),
                         ResidualBlock(filters[1], filters))
            layers.append(res_block)
        
        return nn.Sequential(OrderedDict(layers))
            
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        
        return l3, l4, l5
        
# Res unit       
class ResidualBlock(nn.Module):
    def __init__(self, in_filters, filters):
        super(ResidualBlock, self).__init__()
        
        # DBL 1
        self.conv1 = nn.Conv2d(in_filters, filters[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU(0.1)
        
        # DBL 2
        self.conv2 = nn.Conv2d(filters[0], in_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_filters)
        self.relu2 = nn.ReLU(0.1)



    def forward(self, x):
        
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        # add
        out=out + residual
        
        return out
