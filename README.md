### Self-Guided Network for Fast Image Denoising

This repository is modified based on [the original authors' PyTorch implementation of SGN](https://github.com/zhaoyuzhi/Self-Guided-Network-for-Fast-Image-Denoising). 

`matplotlib` is required to run this repository.

#### What's different compared with the original SGN

This repository modifies the output format of the original one. Specifically, 

- In `train.py`, several arguments are added to the parser `opt`:

  - `debug_str`, decides whether to turn on **debug mode**. It is set to '' if you want to turn OFF debug mode.
  Debug mode is only used when you want to easily get familiar with certain arguments. This string will be added to the name of output path, so that you will know this is a debug-mode run.

  - `dir_path`, indicates the directory **where the output training files are saved**. This argument will be created according to the local time the model starts training, and helps with distinguish different output files.
  The files saved in this path include `.pth` files containing model in certain epochs, an `.png` file of a Loss-Epoch graph, and an `.txt` file consists of all the loss values during the training process.

  - `loss_function`, decides **the loss function used in training**. 
  Specifically, `L1` means to use mean value of the L1 norm, and `MSE` means to use the Mean Squared Error. Note that in `MSE` mode, the output loss will be the Peak Signal-Noise Ratio value for RGB images. For greyscale images, the PSNR value should be multiplied by 3.

  - `crop_randomly`, decides **whether the batches are randomly cropped** from the training dataset. This argument is **ONLY used in debug mode**.
  
- In `trainer.py`, some functions are added to save more training data.

  - `save_loss_data`, `save_loss_graph` and `save_loss_value`. These functions save the Loss-Epoch curve and Loss-Epoch data to the output directory `opt.dir_path`. 

  - `PSNR`, transform the MSE to PSNR value for RGB images.

- In `dataset.py`, a class is modified and a bug is fixed.

  - parameter `randomly` is added to the `__init__` method of class `RandomCrop`. This parameter takes `opt.crop_randomly` as input, and is **ONLY used in debug mode**.

  - a bug is fixed in `__getitem__` method of class `FullResDenoisingDataset`. See `### EM DEBUG` in the code.

- In `validation_folder.py`, the modified part is similar to `train.py` and `trainer.py`. 

- This `test.py` file, which is different from [the original one](https://github.com/zhaoyuzhi/Self-Guided-Network-for-Fast-Image-Denoising/blob/master/SGN/test.py), is derived from `validation_folder.py`. 
