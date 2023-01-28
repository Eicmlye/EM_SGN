import argparse
import os
## EM Modified
import time # for time-based directory name
## end EM Modified

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    # According to paper, the recommend setting is to train the model for 1M iterations, so it is better to save at each 100K iterations
    # The epoch is large enough that the model can be trained more than 1M iterations, users could stop it if it is well trained
    # The learning rate is set to 1e-4 during first 500K iterations, while it is 1e-5 during last 500K iterations
    # For DIV2K dataset: epoch 10000 + batch_size 8 = iteration 1000000; I recommend to save 10 models for the whole training stage
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train or not')
    parser.add_argument('--save_mode', type = str, default = 'iter', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1000, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    ## EM Modified
    parser.add_argument('--dir_path', type = str, default = './RunLocal/', help = 'directory path to save the trained network')
    parser.add_argument('--debug_str', type = str, default = 'debug/debug_', help = 'add \'debug_\' to filename of saved file')
    ## end EM Modified
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 10000, help = 'number of epochs of training that ensures 100K training iterations')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'size of the batches, 8 is recommended')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.9, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--iter_decreased', type = int, default = 500000, help = 'the certain iteration that lr decreased')
    parser.add_argument('--lr_decreased', type = float, default = 0.00001, help = 'decreased learning rate at certain epoch')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    ## EM Modified
    parser.add_argument('--start_epoch', type = int, default = 0, help = 'the training will start at this epoch')
    parser.add_argument('--loss_function', type = str, default = 'L1', help = 'loss function, L1 or MSE')
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
    parser.add_argument('--baseroot', type = str, default = 'C:/Users/yzzha/Desktop/dataset/DIV2K/DIV2K_train_HR', help = 'images baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    ## EM Modified
    parser.add_argument('--validroot', type = str, default = './DIV2K_valid_HR/', help = 'validation set for training')
    parser.add_argument('--crop_randomly', type = bool, default = True, help = 'activate random crop for RandomCrop() in dataset.py')
    ## end EM Modified
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = True, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    
    opt = parser.parse_args()

    '''
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''
    
    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------

    ## EM Modified
    # debug settings
    opt.debug_str = '' # set to '' to turn OFF debug mode
    if opt.debug_str == '':
        print('Debug mode OFF. ')
    else:
        opt.crop_randomly = False
        print('Debug mode ON! ')

    # training settings
    opt.epochs = 100
    opt.save_by_epoch = opt.epochs / 10 # or 1 to save every trained model
    opt.iter_decreased = opt.epochs * 50
    opt.save_mode = 'epoch'
    opt.baseroot = './DIV2K_train_HR/'
    opt.validroot = './DIV2K_valid_HR/'
    opt.loss_function = 'MSE'

    ## if you are gonna continue train pre-trained model, activate this block
    """ # comment this line to activate the block below
    opt.pre_train = False
    opt.load_name = './RunLocal/DataSaved/230124_135613_train10Epochs/DSWN_best_epoch6_bs8_mu0_sigma30.pth'
    
    #"""
    
    # create time-based directory name
    begin_time = time.localtime(time.time())
    opt.dir_path = './RunLocal/' + opt.debug_str + '%02d%02d%02d_%02d%02d%02d_train%dEpochs/' % (begin_time.tm_year - 2000, begin_time.tm_mon, begin_time.tm_mday, begin_time.tm_hour, begin_time.tm_min, begin_time.tm_sec, opt.epochs)
    if not os.path.exists(opt.dir_path):
        os.makedirs(opt.dir_path)
    ## end EM Modified 

    trainer.Trainer(opt)
