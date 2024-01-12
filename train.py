import numpy as np


from nets.yoloV3 import YoloV3
import utils.data_loader as dt_loader
import torch.optim as optim


########### Loading data

# read classes
classes_path    = 'model_data/voc_classes.txt'
class_names, num_classes = dt_loader.get_classes(classes_path)

# model data
input_shape     = [416, 416]
model           = YoloV3()
model_train     = model.train()



# training settings
epoch_total     = 10
batch_size      = 4
learn_rate      = 1e-3
optimizer       = optim.Adam(model_train.parameters(), learn_rate, weight_decay = 5e-4)


# load dataset
annotation_path = "data/VOC2007"
train_dataset, val_dataset = dt_loader.loadData(annotation_path, input_shape, num_classes, batch_size)


for epoch in range(epoch_total):
    
    for iteration, batch in enumerate(train_dataset):
        
        images, tergets = batch[0], batch[1]
        
        # clear gradients
        optimizer.zero_grad()
        
        # forwarding
        outputs = model_train(images)
        
        