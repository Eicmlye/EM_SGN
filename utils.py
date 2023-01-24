import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os
## EM Modified
import matplotlib.pyplot as plt
## end EM Modified

import network

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def create_generator(opt, checkpoint): ## EM ADDED (parameter)checkpoint
    # Initialize the network
    generator = network.SGN(opt)
    if opt.pre_train:
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Load a pre-trained network
        load_dict(generator, checkpoint['net'])
        print('Generator is loaded!')
    return generator

## EM Modified
def create_optimizer(opt, generator, checkpoint):
    optimizer = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    if not opt.pre_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer
    
def save_loss_graph(opt, x, y):
    """
    Save the loss function curve vs. epochs or iterations.

    """

    if opt.multi_gpu == True:
        pass
    else:
        if opt.save_mode == 'epoch':
            # if epoch == opt.epochs: # Save graph only when training is finished
                plt.plot(x, y)
                plt.savefig(opt.dir_path + 'Loss_Epoch.png', dpi=300)
                print('Loss-Epoch graph successfully saved. ')
        else:
            pass

def save_loss_value(opt, y):
    """
    Save the loss value for all epochs.
    
    """
    if opt.multi_gpu == True:
        pass
    else:
        if opt.save_mode == 'epoch':
            if opt.loss_function == 'L1':
                save_path = opt.dir_path + 'L1_Loss_value_Epoch.txt'
            else:
                save_path = opt.dir_path + 'PSNR_value_Epoch.txt'
            file = open(save_path, 'w')

            for epoch in range(len(y)):
                file.write(str(epoch + 1) + '\t:\t' + str(y[epoch]) + '\n')

            file.close()
            print('Loss value successfully saved. ')
        else:
            pass

def save_loss_data(opt, x, y):
    """
    Save the loss graph and loss value for all epochs.
    
    """
    if opt.loss_function == 'MSE':
        y = PSNR(y)

    save_loss_graph(opt, x, y)
    save_loss_value(opt, y)
## end EM Modified
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

## EM Modified
def PSNR(mse): # RGB images, divided by 3 colors
    return 20 * np.log10(255.0 / np.sqrt(mse)) / 3
## end EM Modified
