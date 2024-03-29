## EM Modified
import time # for time-based directory name
import os
import sys
## end EM Modified
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import utils

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    ## EM Modified
    parser.add_argument('--dir_path', type = str, default = './RunLocal/', help = 'directory path to save the trained network')
    ## end EM Modified
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for generator')
    parser.add_argument('--m_block', type = int, default = 2, help = 'the additional blocks used in mainstream')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "/home/alien/Documents/LINTingyu/denoising", help = 'the testing folder')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = float, default = 0, help = 'min scaling factor')
    parser.add_argument('--sigma', type = float, default = 30, help = 'max scaling factor')
    # Other parameters
    parser.add_argument('--pre_train', type = bool, default = False, help = 'test phase')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = 'SGN_iter1000000_bs32_mu0_sigma30.pth', help = 'test model name')
    
    opt = parser.parse_args()
    ## EM deactivated # print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    ## EM Modified
    # set test dataset
    opt.baseroot = './CBSD68/original_png/'

    opt.load_name = './RunLocal/DataSaved/230202_212622_test_CBSD68/SGN_iter1000000_bs32_mu0_sigma30.pth'
    ## end EM Modified

    # Define the dataset
    testset = dataset.FullResDenoisingDataset(opt) # Run full image without cropping
    print('The overall number of images equals to %d. ' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------
    checkpoint = torch.load(opt.load_name)
    model = utils.create_generator(opt, checkpoint).cuda()

    ## EM Modified
    loss_data = [[],[]]

    # create time-based directory name
    opt.dir_path = utils.build_time_based_directory(opt, 'test')
    ## end EM Modified 

    ## EM Note:
    ## as enumerator is created, testset.__getitem__ is called, and RandomCrop is processed
    for img_idx, (noisy_img, img) in enumerate(dataloader): 
        # To Tensor
        noisy_img = noisy_img.cuda()
        img = img.cuda()

        # Generator output
        with torch.no_grad():
            recon_img = model(noisy_img)

        ## EM Modified
        psnr, ssim = utils.PSNR_SSIM_img(img, recon_img)
        loss_data[0].append(psnr)
        loss_data[1].append(ssim)
        ## end EM Modified
        
        # convert to visible image format
        img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)
        noisy_img = noisy_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        noisy_img = (noisy_img + 1) * 128
        noisy_img = noisy_img.astype(np.uint8)
        recon_img = recon_img.data.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        recon_img = (recon_img + 1) * 128
        recon_img = recon_img.astype(np.uint8)

        # show
        show_img = np.concatenate((img, noisy_img, recon_img), axis = 1)
        r, g, b = cv2.split(show_img)
        show_img = cv2.merge([b, g, r])
        cv2.imshow('comparison.jpg', show_img)
        cv2.waitKey(100)
        ## EM Modified: Added time-based directory name
        cv2.imwrite(opt.dir_path + 'pics/result_%04d.jpg' % (img_idx), show_img)

    ## EM Modified
    # save loss data
    save_path = opt.dir_path + 'test_PSNR_SSIM_Epoch.txt'

    file = open(save_path, 'w')

    for picnum in range(len(loss_data[0])):
        file.write(str(picnum + 1) + '\t' + str(loss_data[0][picnum]) + '\t' + str(loss_data[1][picnum]) + '\n')

    file.write('Avg\t' + str(sum(loss_data[0]) / len(loss_data[0])) + '\t' + str(sum(loss_data[1]) / len(loss_data[1])))

    file.close()

    print('Average PSNR: ', sum(loss_data[0]) / len(loss_data[0]), ', average SSIM: ', sum(loss_data[1]) / len(loss_data[1]))
    ## end EM Modified
