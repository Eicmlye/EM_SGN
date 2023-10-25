import numpy as np
import torch
import os
## EM Modified
import matplotlib.pyplot as plt
import time
import pytorch_msssim # https://github.com/VainF/pytorch-msssim
import piqa # https://github.com/francois-rozet/piqa
## end EM Modified

import network

def create_generator(opt, checkpoint): ## EM ADDED (parameter)checkpoint
    # Initialize the network
    generator = network.HWDN(opt)
    if opt.pre_train:
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Load a pre-trained network
        try:
            load_dict(generator, checkpoint['net'])
        except KeyError:
            load_dict(generator, checkpoint)
        print('Generator is loaded!')
    return generator

## EM Modified
def get_time_based_directory(opt, mode = 'train'):
    begin_time = time.localtime(time.time())
    time_path = './RunLocal/%02d%02d%02d_%02d%02d%02d_' % (begin_time.tm_year - 2000, begin_time.tm_mon, begin_time.tm_mday, begin_time.tm_hour, begin_time.tm_min, begin_time.tm_sec)
    dataset_path = ''

    if mode == 'train':
        dataset_path = 'Tot%dEpo_bs%d_mu%d_sigma%d/' % (opt.epochs, opt.batch_size, opt.mu, opt.sigma)
    elif mode == 'test':
        dataset_path = 'test_'

        if opt.baseroot == './CBSD68/original_png/':
            dataset_path += 'CBSD68/'
        elif opt.baseroot == './BSD68/':
            dataset_path += 'BSD68/'
        elif opt.baseroot == './Kodak24/':
            dataset_path += 'Kodak24/'
        elif opt.baseroot == './myTest/':
            dataset_path += 'PureColor/'
        elif opt.baseroot == './DIV2K_train_HR_forTest/':
            dataset_path += 'DIV2Ktrain/'
        elif opt.baseroot == './DIV2K_valid_HR_forTest/':
            dataset_path += 'DIV2Kvalid/'
        else:
            raise ValueError('New test dataset. Please add save path to utils.py/get_time_based_directory()')

    return time_path + dataset_path

def build_directory(path, mode = 'train'):
    if not os.path.exists(path):
        os.makedirs(path)
    if mode == 'train':
        if not os.path.exists(path + 'models/'):
            os.makedirs(path + 'models/')
        if not os.path.exists(path + 'best_models/'):
            os.makedirs(path + 'best_models/')
    elif mode == 'test':
        if not os.path.exists(path + 'pics/'):
            os.makedirs(path + 'pics/')

def build_time_based_directory(opt, mode = 'train'):
    dir_name = get_time_based_directory(opt, mode)

    build_directory(dir_name, mode)

    return dir_name        

def create_optimizer(opt, generator, checkpoint):
    optimizer = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    if not opt.pre_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer

def load_loss_data(cur_epoch, load_loss_name):
    '''
    loss data file format: epoch_num    PSNR    SSIM
    '''
    f = open(load_loss_name, 'r')

    y = [[], []]
    for line in f:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        y[0].append(eval(words[1]))
        y[1].append(eval(words[2]))

        if len(y[0]) == cur_epoch:
            break

    f.close()

    return y
    
def save_loss_graph(opt, y):
    """
    Save the loss function curve vs. epochs or iterations.

    """
    ## plt.figure(figsize(width_base, height_base))
    ## plt.savefig(path, dpi=DPI)
    ## The overall graph size is (width_base * DPI) * (height_base * DPI)
    plt.figure(figsize=(36.2, 10)) # 1920:1440 = 1.81:1

    ## plot PSNR graph
    # np.linspace(lowerbound, upperbound, bound_dist * (ticks_in_open_interval_of_length_one + 1) + 1)
    # np.linspace(lowerbound, upperbound, (bound_dist + 1) // step + 1)
    # np.linspace(lowerbound, upperbound, total_num_of_ticks)
    xticks = np.linspace(1, opt.epochs, (opt.epochs - 1 + 1) // 20 + 1, dtype=int)
    yticks = np.linspace(18, 26, (26 - 18) * 1 + 1)
    yticks = np.concatenate((yticks, np.linspace(27, 38, (38 - 27) * 2 + 1)))

    # initialize PSNR graph
    plt.subplot(1, 2, 1)
    plt.title('PSNR vs. Epochs')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')

    x = [i for i in range(1, len(y[0]) + 1)]

    plt.plot(x, y[0], color='r', linestyle='-', label='PSNR')

    ## plot SSIM graph
    # np.linspace(lowerbound, upperbound, bound_dist * (ticks_in_open_interval_of_length_one + 1) + 1)
    # np.linspace(lowerbound, upperbound, (bound_dist + 1) // step + 1)
    # np.linspace(lowerbound, upperbound, total_num_of_ticks)
    xticks = np.linspace(1, opt.epochs, (opt.epochs - 1 + 1) // 20 + 1, dtype=int)
    yticks = np.linspace(0, 1, (1 - 0) * 20 + 1)

    # initialize SSIM graph
    plt.subplot(1, 2, 2)
    plt.title('SSIM vs. Epochs')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')

    plt.plot(x, y[1], color='b', linestyle='-', label='SSIM')

    ## save graph
    plt.savefig(opt.dir_path + 'PSNR_SSIM_Epoch.png', dpi=300)
    print('Loss-Epoch graph successfully saved. ')

    plt.close()

def save_loss_value(opt, y):
    """
    Save the loss value for all epochs.
    epoch_num   psnr    ssim
    """

    if opt.multi_gpu == True:
        pass
    else:
        if opt.save_mode == 'epoch':
            save_path = opt.dir_path + 'PSNR_SSIM_value_Epoch.txt'
            file = open(save_path, 'w')

            for epoch in range(len(y[0])):
                file.write(str(epoch + 1) + '\t' + str(y[0][epoch]) + '\t' + str(y[1][epoch]) + '\n')

            file.close()
            print('Loss value successfully saved. ')
        else:
            pass

def save_loss_data(opt, y):
    """
    Save the loss graph and loss value for all epochs.
    
    """

    save_loss_graph(opt, y)
    save_loss_value(opt, y)

"""
def PSNR(mse: float):
    return 20 * np.log10(255.0 / np.sqrt(mse))

def SSIM(avg_img, avg_recon_img, var_img, var_recon_img, covar, max_pixel_value = 255):
    k1 = 0.01
    k2 = 0.03
    L = max_pixel_value

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    return (2 * avg_img * avg_recon_img + c1) * (2 * covar + c2) / (avg_img ** 2 + avg_recon_img ** 2 + c1) / (var_img ** 2 + var_recon_img ** 2 + c2)

def denormalize(img: torch.Tensor, recon_img: torch.Tensor):
    img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) * 128
    img = img.astype(np.float32)
    
    recon_img = recon_img.data.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    recon_img = (recon_img + 1) * 128
    recon_img = recon_img.astype(np.float32)

    return img, recon_img
"""

def PSNR_SSIM_img(img: torch.Tensor, recon_img: torch.Tensor):
    # for PSNR
        # notice that the dimension of img and noisy_img
        # is opt.batch_size * color_channel_num * opt.crop_size * opt.crop_size.
    """
    batch_size = img.shape[0]
    psnr = 0
    
    # for SSIM
    ssim = 0
    
    # computation
    for i in range(batch_size):
        cache_img, cache_recon_img = denormalize(img[i], recon_img[i])

        # for PSNR
        mse = np.mean((cache_recon_img - cache_img)**2) # MSE of r, g, b channels
        psnr += PSNR(mse)

        # for SSIM
        avg_img = np.mean(cache_img)
        avg_recon_img = np.mean(cache_recon_img)
        var_img = np.sqrt(np.mean((cache_img - avg_img)**2))
        var_recon_img = np.sqrt(np.mean((cache_recon_img - avg_recon_img)**2))
        covar = np.mean((cache_img - avg_img) * (cache_recon_img - avg_recon_img))
        ssim += SSIM(avg_img, avg_recon_img, var_img, var_recon_img, covar, 255)
        
    psnr /= batch_size
    ssim /= batch_size
    """
    # for piqa psnr
    """
    print(img.shape)
    print(piqa.psnr.psnr((img + 1) * 128, (recon_img + 1) * 128, value_range=255).shape)
    input()
    """
    psnr = torch.mean(piqa.psnr.psnr((img + 1) * 128, (recon_img + 1) * 128, value_range=255)).item()
    # for pytorch_ssim
    ssim = pytorch_msssim.ssim((img + 1) * 128, (recon_img + 1) * 128).item()

    return psnr, ssim
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

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret
