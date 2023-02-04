import time
import datetime
import numpy as np
## EM Modified
import matplotlib.pyplot as plt
import sys # for exit()
## end EM Modified
import torch
import torch.nn as nn
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
    loss_criterion = torch.nn.L1Loss().cuda()

    ## EM Modified
    y = [[],[]]

    # load checkpoint info
    if opt.pre_train:
        checkpoint = {}
        best_psnr = -1
        best_ssim = -1
    else:
        checkpoint = torch.load(opt.load_name)
        opt.start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        best_ssim = checkpoint['best_ssim']
        y = utils.load_loss_data(opt.load_loss_name)
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

    del checkpoint
    
    # Learning rate decrease
    def adjust_learning_rate(opt, iteration, optimizer):
        # Set the learning rate to the specific value
        if iteration >= opt.iter_decreased[0]:
            for i in range(len(opt.iter_decreased) - 1):
                if iteration >= opt.iter_decreased[i] and iteration < opt.iter_decreased[i + 1]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = opt.lr_decreased[i]
                    break
            
            if iteration >= opt.iter_decreased[-1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.lr_decreased[-1]

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, network, optimizer, best_psnr, best_ssim):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(network.module.state_dict(), opt.dir_path + 'models/SGN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at epoch %d. ' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(network.module.state_dict(), opt.dir_path + 'models/SGN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at iteration %d. ' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                checkpoint = {'epoch':epoch, 'best_psnr':best_psnr, 'best_ssim':best_ssim, 'net':network.state_dict(), 'optimizer':optimizer.state_dict()}
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(checkpoint, opt.dir_path + 'models/SGN_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at epoch %d. ' % (epoch))
            if opt.save_mode == 'iter':
                checkpoint = {'iteration':iteration, 'best_psnr':best_psnr, 'best_ssim':best_ssim, 'net':network.state_dict(), 'optimizer':optimizer.state_dict()}
                if iteration % opt.save_by_iter == 0:
                    torch.save(checkpoint, opt.dir_path + 'models/SGN_iter%d_bs%d_mu%d_sigma%d.pth' % (iteration, opt.batch_size, opt.mu, opt.sigma))
                    print('The trained model is successfully saved at iteration %d. ' % (iteration))

    def save_best_model(opt, psnr, ssim, best_psnr, best_ssim, epoch, network, optimizer):
        if best_psnr < psnr and best_ssim < ssim and epoch >= opt.epochs / 10:
            best_psnr = psnr
            best_ssim = ssim

            checkpoint = {'epoch':epoch, 'best_psnr':best_psnr, 'best_ssim':best_ssim, 'net':network.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(checkpoint, opt.dir_path + 'best_models/SGN_best_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
            print('The best model is successfully updated. This model is the best one in both PSNR and SSIM. ')
        else:
            if best_psnr < psnr and epoch >= opt.epochs / 10:
                best_psnr = psnr

                checkpoint = {'epoch':epoch, 'best_psnr':best_psnr, 'best_ssim':best_ssim, 'net':network.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(checkpoint, opt.dir_path + 'best_models/SGN_best_psnr_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                print('The best PSNR model is successfully updated. ')
            if best_ssim < ssim and epoch >= opt.epochs / 10:
                best_ssim = ssim

                checkpoint = {'epoch':epoch, 'best_psnr':best_psnr, 'best_ssim':best_ssim, 'net':network.state_dict(), 'optimizer':optimizer.state_dict()}
                torch.save(checkpoint, opt.dir_path + 'best_models/SGN_best_ssim_epoch%d_bs%d_mu%d_sigma%d.pth' % (epoch, opt.batch_size, opt.mu, opt.sigma))
                print('The best SSIM model is successfully updated. ')

        return best_psnr, best_ssim

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DenoisingDataset(opt)
    # validset = dataset.FullResDenoisingDataset(opt, opt.validroot)
    validset = dataset.DenoisingDataset(opt, opt.validroot)
    len_valid = len(validset)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    validloader = DataLoader(validset, batch_size = 1, pin_memory = True)
    del trainset, validset

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    validation_time = datetime.timedelta(seconds=30)

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
            del noisy_img

            loss = loss_criterion(recon_img, img) ## EM Modified
            psnr_data, ssim_data = utils.PSNR_SSIM_img(img, recon_img)
            del img, recon_img

            # Overall Loss and optimize
            loss.backward()
            del loss
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = validation_time * (opt.epochs - epoch) + datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d]\t[Batch %d/%d]\t[Recon PSNR: %.4f]\t[Recon SSIM: %.4f]\tTime_left: %s" % ((epoch + 1), opt.epochs, (i + 1), len(dataloader), psnr_data, ssim_data, str(time_left)[:-7]))

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (iters_done + 1), optimizer_G)
        
        ## EM Modified
        # validation
        print('---- Validation ----')
        print('The overall number of validation images: ', len_valid)

        psnr_avg = 0
        ssim_avg = 0

        for j, (noisy_img, img) in enumerate(validloader):
            # To device
            noisy_img = noisy_img.cuda()
            img = img.cuda()

            # Forward propagation
            with torch.no_grad():
                recon_img = generator(noisy_img)
            del noisy_img
            
            loss = loss_criterion(recon_img, img)
            psnr_data, ssim_data = utils.PSNR_SSIM_img(img, recon_img)
            del img, recon_img, loss

            psnr_avg += psnr_data
            ssim_avg += ssim_data

            # Print log
            print("\rEpoch %d\t[Image %d/%d]\t[Recon PSNR: %.4f]\t[Reson SSIM: %.4f]" %
                    ((epoch + 1), (j + 1), len(validloader), psnr_data, ssim_data))

        psnr_avg /= len(validloader)
        ssim_avg /= len(validloader)
        print("Average PSNR for validation set: %.2f" % (psnr_avg))
        print("Average SSIM for validation set: %.4f" % (ssim_avg))

        # save loss graph
        if opt.save_mode == 'epoch':
            y[0].append(psnr_avg)
            y[1].append(ssim_avg)
            utils.save_loss_data(opt, y)
        else:
            pass

        # update best loss and best model
        best_psnr, best_ssim = save_best_model(opt, psnr_avg, ssim_avg, best_psnr, best_ssim, (epoch + 1), generator, optimizer_G)
        # Save model at certain epochs or iterations
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator, optimizer_G, best_psnr, best_ssim)
        
        validation_time = datetime.timedelta(seconds=time.time() - prev_time)
        prev_time = time.time()
        ## end EM Modified
