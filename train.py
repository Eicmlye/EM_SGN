import argparse

import trainer
import utils

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
        # According to paper, the recommend setting is to train the model 
        # for 1M iterations, so it is better to save at each 100K iterations
        # The epoch is large enough that the model can be trained 
        # more than 1M iterations, users could stop it if it is well trained
        # The learning rate is set to 1e-4 during first 500K iterations, 
        # while it is 1e-5 during last 500K iterations
        # For DIV2K dataset: epoch 10000 + batch_size 8 = iteration 1000000; 
        # I recommend to save 10 models for the whole training stage
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train or not')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1000, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 100000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    ## EM Modified
    parser.add_argument('--load_loss_name', type = str, default = '', help = 'load the loss data of pre-trained model')
    parser.add_argument('--dir_path', type = str, default = './RunLocal/', help = 'directory path to save the trained network')
    ## end EM Modified

    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0,1,2,3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')

    # Training parameters
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training that ensures 100K training iterations')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr', type = float, default = 0.0001 * 1, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--iter_decreased', type = list, default = [200 * 800, 250 * 800], help = 'the certain iteration that lr decreased')
    parser.add_argument('--lr_decreased', type = list, default = [0.00005 * 1, 0.00001 * 1], help = 'decreased learning rate at certain epoch')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    ## EM Modified
    parser.add_argument('--start_epoch', type = int, default = 0, help = 'the training will start at this epoch index')
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
    parser.add_argument('--baseroot', type = str, default = 'C:/Users/yzzha/Desktop/dataset/DIV2K/DIV2K_train_HR', help = 'images baseroot')
    ## EM Modified
    parser.add_argument('--validroot', type = str, default = './DIV2K_valid_HR/', help = 'validation set for training')
    ## end EM Modified
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
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
    # training settings
    opt.epochs = 300
    opt.save_by_epoch = 10 # or set to opt.epoch to save every trained model
    opt.baseroot = './DIV2K_train_HR/'
    opt.validroot = './DIV2K_valid_HR/'

    ## EM COMMENT: my GPU memory is too small for 256 crop_size and 1 batch_size
    opt.crop_size = 128
    opt.batch_size = 2

    lr_inc = 1
    opt.lr = 0.00005 * lr_inc
    opt.iter_decreased = [200, 250]
    opt.iter_decreased = [i * 800 // opt.batch_size for i in opt.iter_decreased]
    opt.lr_decreased = [0.00001, 0.000005]
    opt.lr_decreased = [i * lr_inc for i in opt.lr_decreased]

    ## if you are gonna continue train pre-trained model, activate this block
    """ # comment this line to activate the block below
    opt.pre_train = False

    load_base = './RunLocal/230203_122755_Tot300Epo_bs32_mu0_sigma30/'
    opt.load_name = load_base + 'models/DSWN_epoch1_bs32_mu0_sigma30.pth'
    opt.load_loss_name = load_base + 'PSNR_SSIM_value_Epoch.txt'
    #"""
    
    # create time-based directory
    opt.dir_path = utils.build_time_based_directory(opt)
    ## end EM Modified 

    print('Total epochs: %d.\tModels will be saved every %d epochs.' % (opt.epochs, opt.save_by_epoch))
    print('Batch size: %d.\t\tCrop size: %d.' % (opt.batch_size, opt.crop_size))

    trainer.Trainer(opt)
