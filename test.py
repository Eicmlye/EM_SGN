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
    ## EM Modified
    def PSNR(mse): # RGB images, divided by 3 colors
        return 20 * np.log10(255.0 / np.sqrt(mse)) / 3
    ## end EM Modified

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    ## EM Modified
    parser.add_argument('--dir_path', type = str, default = './RunLocal/', help = 'directory path to save the trained network')
    parser.add_argument('--debug_str', type = str, default = 'debug/debug_', help = 'add \'debug_\' to filename of saved file')
    ## end EM Modified
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'input channels for generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for generator')
    parser.add_argument('--m_block', type = int, default = 2, help = 'the additional blocks used in mainstream')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "/home/alien/Documents/LINTingyu/denoising", help = 'the testing folder')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    ## EM Modified
    parser.add_argument('--crop_randomly', type = bool, default = False, help = 'test needs to process full image, hence deactivate random crop for RandomCrop() in dataset.py')
    parser.add_argument('--dataset', type = str, default = 'DIV2K', help = 'dataset used for test, DIV2K or BSD68')
    ## end EM Modified
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
    opt.debug_str = '' # set to '' to turn OFF debug mode
    if opt.debug_str == '':
        print('Debug mode OFF. ')
    else:
        print('Debug mode ON! ')

    # set test dataset
    opt.dataset = 'BSD68'
    if opt.dataset == 'DIV2K':
        opt.baseroot = './DIV2K_valid_HR/'
    elif opt.dataset == 'BSD68':
        opt.baseroot = './CBSD68/original_png/'
    else:
        pass # Unknown dataset, use opt.baseroot

    opt.load_name = './RunLocal/230123_225954_train10Epochs/SGN_epoch4_bs8_mu0_sigma30.pth'

    opt.loss_function = 'MSE'
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
    loss_data = []

    # create time-based directory name
    begin_time = time.localtime(time.time())
    opt.dir_path = './RunLocal/' + opt.debug_str + '%02d%02d%02d_%02d%02d%02d_test_' \
        % (begin_time.tm_year - 2000, begin_time.tm_mon, begin_time.tm_mday, \
        begin_time.tm_hour, begin_time.tm_min, begin_time.tm_sec)\
        + opt.dataset + '/'
    if not os.path.exists(opt.dir_path):
        os.makedirs(opt.dir_path)
    ## end EM Modified 

    ## EM Note:
    ## as enumerator is created, testset.__getitem__ is called, and RandomCrop is processed
    for img_idx, (noisy_img, img) in enumerate(dataloader): 
        # To Tensor
        noisy_img = noisy_img.cuda()
        img = img.cuda()
        ## EM Modified
        # Loss functions ## EM Added MSE
        if opt.loss_function == 'L1':
            loss_criterion = torch.nn.L1Loss().cuda()
        elif opt.loss_function == 'MSE':
            loss_criterion = torch.nn.MSELoss().cuda()
        else:
            print('Unknown loss criterion. ')
            sys.exit()
        ## end EM Modified

        # Generator output
        with torch.no_grad():
            recon_img = model(noisy_img)

        ## EM Modified
        loss = loss_criterion(recon_img, img)
        loss_data.append(loss.item())
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
        if opt.dataset == 'DIV2K':
            cv2.imwrite(opt.dir_path + 'result_%04d.jpg' % (img_idx + 801), show_img)
        else:
            cv2.imwrite(opt.dir_path + 'result_%04d.jpg' % (img_idx), show_img)

    ## EM Modified
    # save loss data
    if opt.loss_function == 'L1':
        save_path = opt.dir_path + 'test_L1_Loss_value_Epoch.txt'
    else:
        save_path = opt.dir_path + 'test_PSNR_value_Epoch.txt'
        loss_data = PSNR(loss_data)
    file = open(save_path, 'w')

    for picnum in range(len(loss_data)):
        file.write(str(picnum + 1))
        file.write('\t:\t')
        file.write(str(loss_data[picnum]))
        file.write('\n')

    file.write('Avg\t:\t')
    file.write(str(sum(loss_data) / len(loss_data)))

    file.close()
    ## end EM Modified
