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

    ## EM Modified
    # initialize loss graph
    x = []
    y = []
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    if opt.loss_function == 'MSE':
        plt.ylabel('PSNR')
    else:
        plt.ylabel('L1 Loss')
    plt.ion() # activate interactive mode
    
    # save best loss value
    best_loss = 10000

    # load checkpoint info
    if opt.pre_train:
        checkpoint = {}
    else:
        checkpoint = torch.load(opt.load_name)
        opt.start_epoch = checkpoint['epoch']
        x = [i for i in range(1, opt.start_epoch)]
        y = checkpoint['loss_data']
        best_loss = checkpoint['best_loss']
    ## end EM Modified

    # Initialize SGN
    generator = utils.create_generator(opt, checkpoint)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = utils.create_optimizer(opt, generator, checkpoint)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, iteration, optimizer):
        # Set the learning rate to the specific value
        if iteration >= opt.iter_decreased:
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_decreased

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, network, optimizer, y, best_loss):
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
                checkpoint = {'epoch':epoch, 'net':network.state_dict(), 'optimizer':optimizer.state_dict(), 'loss_data':y, 'best_loss':best_loss}
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(checkpoint, opt.dir_path + 'SGN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at epoch %d. ' % (epoch))
            if opt.save_mode == 'iter':
                checkpoint = {'iteration':iteration, 'net':network.state_dict(), 'optimizer':optimizer.state_dict(), 'loss_data':y, 'best_loss':best_loss}
                if iteration % opt.save_by_iter == 0:
                    torch.save(checkpoint, opt.dir_path + 'SGN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at iteration %d. ' % (iteration))

    def save_best_model(opt, loss, best_loss, epoch, network, optimizer, y):
        if epoch > opt.epochs / 2 and best_loss > loss:
            best_loss = loss

            checkpoint = {'epoch':epoch, 'net':network.state_dict(), 'optimizer':optimizer.state_dict(), 'loss_data':y, 'best_loss':best_loss}
            torch.save(checkpoint, opt.dir_path + 'SGN_best_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
            print('The best model is successfully updated. ')
        return best_loss

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DenoisingDataset(opt)
    ## EM COMMENT: if FullRes is used, the validloader batch_size should set to 1. 
    # validset = dataset.FullResDenoisingDataset(opt, opt.validroot)
    validset = dataset.DenoisingDataset(opt, opt.validroot)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    ## EM COMMENT: if FullRes is used, the validloader batch_size should set to 1. 
    # validloader = DataLoader(validset, batch_size = 1, pin_memory = True)
    validloader = DataLoader(validset, batch_size = 1, pin_memory = True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.start_epoch, opt.epochs):
        print('\n==== Epoch %d below ====\n' % (epoch + 1))

        for i, (noisy_img, img) in enumerate(dataloader):
            # To device
            noisy_img = noisy_img.cuda()
            img = img.cuda()

            # Train Generator
            optimizer_G.zero_grad()

            # Forward propagation
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
            if (opt.loss_function == 'MSE'):
                print("\r[Epoch %d/%d]\t[Batch %d/%d]\t[Recon Loss: %.4f]\tTime_left: %s" %
                    ((epoch + 1), opt.epochs, (i + 1), len(dataloader), utils.PSNR(loss.item()), str(time_left)[:-7]))
            else:
                print("\r[Epoch %d/%d]\t[Batch %d/%d]\t[Recon Loss: %.4f]\tTime_left: %s" %
                    ((epoch + 1), opt.epochs, (i + 1), len(dataloader), loss.item(), str(time_left)[:-7]))

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (iters_done + 1), optimizer_G)
        
        ## EM Modified
        # validation
        print('---- Validation ----')
        print('The overall number of validation images: ', len(validset))

        loss_avg = 0

        for j, (noisy_valimg, valimg) in enumerate(validloader):
            # To device
            noisy_valimg = noisy_valimg.cuda()
            valimg = valimg.cuda()

            # Forward propagation
            with torch.no_grad():
                recon_valimg = generator(noisy_valimg)
            valloss = loss_criterion(recon_valimg, valimg)
            loss_avg += valloss.item()

            # Print log
            if (opt.loss_function == 'MSE'):
                print("\rEpoch %d\t[Image %d/%d]\t[Recon Loss: %.4f]" %
                    ((epoch + 1), (j + 1), len(validloader), utils.PSNR(valloss.item())))
            else:
                print("\rEpoch %d\t[Image %d/%d]\t[Recon Loss: %.4f]" %
                    ((epoch + 1), (j + 1), len(validloader), valloss.item()))

        loss_avg /= len(validloader)
        if (opt.loss_function == 'MSE'):
            print("Average PSNR for validation set: %.2f" % (utils.PSNR(loss_avg)))
        else:
            print("Average loss for validation set: %.2f" % (loss_avg))

        # Save model at certain epochs or iterations
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator, optimizer_G, y, best_loss)
        # update best loss and best model
        best_loss = save_best_model(opt, loss_avg, best_loss, (epoch + 1), generator, optimizer_G, y)

        # save loss graph
        if opt.save_mode == 'epoch':
            x.append(epoch + 1)
            y.append(loss_avg)
            utils.save_loss_data(opt, x, y)
        else:
            pass
        ## end EM Modified
