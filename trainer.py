import time
import datetime
import itertools
import numpy as np
## EM Modified
import matplotlib.pyplot as plt
import sys # for exit()
## end EM Modified
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions ## EM Added MSE
    if opt.loss_function == 'L1':
        loss_criterion = torch.nn.L1Loss().cuda()
    elif opt.loss_function == 'MSE':
        loss_criterion = torch.nn.MSELoss().cuda()
    else:
        print('Unknown loss criterion. ')
        sys.exit()

    # Initialize SGN
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, iteration, optimizer):
        # Set the learning rate to the specific value
        if iteration >= opt.iter_decreased:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_decreased

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, network):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(network.module.state_dict(), opt.dir_path + 'SGN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at epoch %d. ' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(network.module.state_dict(), opt.dir_path + 'SGN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at iteration %d. ' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(network.state_dict(), opt.dir_path + 'SGN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at epoch %d. ' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(network.state_dict(), opt.dir_path + 'SGN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at iteration %d. ' % (iteration))

    ## EM Modified
    def PSNR(mse): # RGB images, divided by 3 colors
        return 20 * np.log10(255.0 / np.sqrt(mse)) / 3

    def save_loss_graph(opt, x, y, epoch, iteration):
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
                    file.write(str(epoch))
                    file.write('\t:\t')
                    file.write(str(y[epoch]))
                    file.write('\n')

                file.close()
                print('Loss value successfully saved. ')
            else:
                pass

    def save_loss_data(opt, x, y, epoch, iteration):
        """
        Save the loss graph and loss value for all epochs.
        
        """
        if opt.loss_function == 'MSE':
            y = PSNR(y)

        save_loss_graph(opt, x, y, epoch, iteration)
        save_loss_value(opt, y)
    ## end EM Modified


    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DenoisingDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    ## EM Modified
    x = []
    y = []
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    if opt.loss_function == 'MSE':
        plt.ylabel('PSNR')
    else:
        plt.ylabel('L1 Loss')
    plt.ion() # activate interactive mode
    ## end EM Modified

    # For loop training
    for epoch in range(opt.epochs):
        print('\n==== Epoch %d below ====\n' % (epoch + 1))

        for i, (noisy_img, img) in enumerate(dataloader):

            # To device
            noisy_img = noisy_img.cuda()
            img = img.cuda()

            # Train Generator
            optimizer_G.zero_grad()

            # Forword propagation
            recon_img = generator(noisy_img)
            loss = loss_criterion(recon_img, img) ## EM Modified

            # Overall Loss and optimize
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d]\t[Batch %d/%d]\t[Recon Loss: %.4f]\tTime_left: %s" %
                ((epoch + 1), opt.epochs, (i + 1), len(dataloader), PSNR(loss.item()), str(time_left)[:-7]))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (iters_done + 1), optimizer_G)
        
        ## EM Modified: save loss graph
        if opt.save_mode == 'epoch':
            x.append(epoch + 1)
            y.append(loss.item())
            save_loss_data(opt, x, y, epoch + 1, 0)
        else:
            pass
        ## end EM Modified
