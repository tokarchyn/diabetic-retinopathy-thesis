# Detecting Diabetic Retinopathy and Related Diagnoses Using Neural Networks

In this project we apply convolutional neural network to classify diabetic retinopathy into 5 classes.  

## Instruction
- Install python environment
- Install NVIDIA CUDA Toolkit. [Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install python packages: `pip install -r requirements.txt`
- Download and unzip dataset from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
- Preprocess images with `preprocess.py` script
- Train model using `train_with_cli.py` script

##### Usage: preprocess.py
Methods to preprocess:
- winner - the method used by winner in competition on [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
- my_gauss - extension of winner
- clahe - the same as my_gauss but instead of substraction Gaussian filtered image, CLAHE is used  
```
  -h, --help            show this help message and exit
  --input INPUT         directory with original images
  --output OUTPUT       directory to put processed images
  --method {my_gauss,winner,clahe}
                        method for preprocess. Default is my_gauss
```

##### Usage: train_with_cli.py
```
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        directory path with images
  --dataframe_path DATAFRAME_PATH
                        path to .csv file with image labels
  --quality_dataset_path QUALITY_DATASET_PATH
                        path to .csv file with quality of images. From 0 (the
                        best) to 2(the worse). Images with the quality level
                        of 2 will be skipped
  --experiments_dir EXPERIMENTS_DIR
                        directory path to where plots and model weights will
                        be saved
  --gpu_id GPU_ID       id of gpu to use
  --model MODEL         architecture of model. Available options: vgg,
                        inception, alex, all_cnn, inception_clean, efficient
  --img_size IMG_SIZE   images will be resized to this size
  --batch_size BATCH_SIZE
  --cyclic_lr           use cyclic learning rate
  --optimizer OPTIMIZER
                        optimizer. Available options: adam, sgd, rmsprop
  --activation ACTIVATION
                        activation function. Available options: leaky_relu,
                        relu
  --learning_rate LEARNING_RATE
  --momentum MOMENTUM   momentum. Will be used only if optimizer is set to sgd
  --augment             enable augmentation
  --bias_reg            enable bias L2 regularization
  --kernel_reg          enable kernel L2 regularization
  --balance_mode BALANCE_MODE
                        resample mode. Max - oversampling every class to reach
                        majority one. Min - undersampling each class to
                        minority one. Integer value - get n from each class
  --checkpoint_path CHECKPOINT_PATH
                        path to file with weights for model
```