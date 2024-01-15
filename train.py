import numpy as np
import time

from nets.yoloV3 import YoloV3, weights_init
import utils.data_loader as dt_loader
import torch.optim as optim
import torch
import torch.backends.cudnn as cudnn


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    
    Cuda = True

    ########### Loading data

    # read classes
    classes_path    = 'model_data/voc_classes.txt'
    class_names, num_classes = dt_loader.get_classes(classes_path)

    # model data
    input_shape     = [416, 416]
    model           = YoloV3(num_classes, input_shape)
    weights_init(model)
    
    model_train     = model.train()
    if Cuda:
        model_train     = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train     = model_train.cuda()


    # training settings
    epoch_total     = 1
    batch_size      = 2
    learn_rate      = 1e-3
    optimizer       = optim.Adam(model_train.parameters(), learn_rate, weight_decay = 5e-4)


    # load dataset
    annotation_path = "data/VOC2007"
    train_dataset, val_dataset = dt_loader.loadData(annotation_path, input_shape, num_classes, batch_size)

    loss = 0.0
    for epoch in range(epoch_total):
        print("----------------------epoch[%d]----------------------" % (epoch))
        for iteration, batch in enumerate(train_dataset):
            
            images, GTs = batch[0], batch[1]
            with torch.no_grad():
                if Cuda:
                    # images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    GTs = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in GTs]
                else:
                    # images  = torch.from_numpy(images).type(torch.FloatTensor)
                    GTs = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in GTs]
                    
            # clear gradients
            optimizer.zero_grad()
            
            # forwarding
            t_0 = time.time()
            tensor_NCHW = torch.tensor(dt_loader.conversion_NCWH(images)).cuda()
            outputs = model_train(tensor_NCHW)
            t_1 = time.time()
            # calculate loss for each head
            loss = model.loss(outputs, GTs)
            t_2 = time.time()
            print("forwarding time: %.3f | computing loss time: %.3f ->loss %.3f"%(t_1-t_0, t_2-t_1, loss))
            
            # BP
            loss.backward()
            
            optimizer.step()
            
            
    print("training is finished")
    torch.save(model.state_dict(),'model_data/ep%03d-loss%.3f.pth' % (epoch + 1, loss))
    
        
            
            
        