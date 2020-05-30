import tensorflow as tf
from tensorflow.keras.optimizers import *

import json
import argparse
from pathlib import Path
import datetime
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.ioff()

from modules.data_loader import create_datasets
from modules.cyclic_lr import CyclicLR
from modules.f_score import F1Score
from modules.cohen_kappa import CohenKappa
from modules.plots import create_confusion_matrix
from modules.augmenter import augment
from modules.training_history_callback import TrainingHistoryCallback
from models.alex_model import get_alex_model
from models.all_cnn_model import get_all_cnn_model
from models.inception_v3_model import get_inception_v3
from models.inception_v3_clean_model import get_inception_v3_clean
from models.efficient_model import get_efficient
from models.vgg_16_model import get_vgg_model

# %precision % .5f
np.set_printoptions(suppress=True, precision=5)

# Define constants
CLASS_NAMES = np.array(
    ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
CLASS_INDEXES = [0, 1, 2, 3, 4]


def init_env():
    args = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='train_processed')
    parser.add_argument('--dataframe_path', type=str,
                        default='data/train_labels_full.csv')
    parser.add_argument('--quality_dataset_path',
                        type=str, default=None)
    parser.add_argument('--experiments_dir', type=str,
                        default='experiments')
    parser.add_argument('--gpu_id', type=str,
                        default=None)
    parser.add_argument('--model', type=str,
                        default='vgg')
    parser.add_argument('--img_size', type=int,
                        default=512)
    parser.add_argument('--batch_size', type=int,
                        default=16)

    parser.add_argument('--cyclic_lr', action='store_true')
    parser.add_argument('--optimizer', type=str,
                        default='adam')
    parser.add_argument('--activation', type=str,
                        default='leaky_relu')
    parser.add_argument('--learning_rate', type=float,
                        default=0.000005)
    parser.add_argument('--momentum', type=float,
                        default=0.9)

    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--bias_reg', action='store_true')
    parser.add_argument('--kernel_reg', action='store_true')
    parser.add_argument('--balance_mode', type=str,
                        default=None)
    parser.add_argument('--checkpoint_path', type=str,
                        default=None)
    parsed_args = parser.parse_args()
    args = vars(parsed_args)

    if args['gpu_id'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu_id']
        tf_device='/gpu:0'

    print('Arguments:', json.dumps(args))
    return args


def create_experiment(params, args):
    experiment_dir = os.path.join(args['experiments_dir'],
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    params_to_log = {k: v if is_jsonable(v) else str(v) for k, v in {**params, **args}.items()}
    print(json.dumps(params_to_log), 
        file=open(os.path.join(experiment_dir, 'params.json'), 'w+'))
    return experiment_dir


def get_input_shape(img_size):
    return (img_size, img_size, 3)

def get_training_params(args):
    params = {
        'input_shape': get_input_shape(args['img_size']),
        'class_number': len(CLASS_NAMES),
        'metrics': ['accuracy', F1Score(len(CLASS_INDEXES)), CohenKappa(len(CLASS_INDEXES))],
        'kernel_reg': tf.keras.regularizers.l2(5e-4) if args['kernel_reg'] else None,
        'bias_reg': tf.keras.regularizers.l2(5e-4) if args['bias_reg'] else None
    }

    if args['optimizer'] == 'adam':
        params['optimizer'] = Adam(lr=args['learning_rate'])
    elif args['optimizer'] == 'sgd':
        params['optimizer'] = SGD(lr=args['learning_rate'], momentum=args['momentum'])

    if args['activation'] == 'leaky_relu':
        params['activation'] = tf.keras.layers.LeakyReLU(alpha=0.3)
    elif args['activation'] == 'relu':
        params['activation'] = tf.keras.layers.ReLU()

    return params


# Callbacks and metrics
def get_callbacks(save_best_models=True, best_models_dir=None,
                  early_stopping=True,
                  reduce_lr_on_plateau=False,
                  training_history=True, metrics_plot_dir=None,
                  cyclic_lr=True, cyclic_lr_step_size=-1,
                  class_names=[],
                  learning_rate=0.001):
    callbacks = []
    if save_best_models:
        Path(best_models_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                best_models_dir, 'model.hdf5'),
            monitor='val_cohen_kappa',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='max'))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_cohen_kappa',
            patience=200,
            mode='max',
            restore_best_weights=True))
    if reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=10,
            verbose=1,
            mode='auto',
            epsilon=0.0001,
            cooldown=5,
            min_lr=0.00001))
    if training_history:
        Path(metrics_plot_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(TrainingHistoryCallback(
            ['loss', 'accuracy', 'f1_score', 'lr', 'cohen_kappa'],
            metrics_plot_dir = metrics_plot_dir,
            save_weights = False,
            class_names = class_names))
    if cyclic_lr:
        callbacks.append(CyclicLR(
            mode='triangular2',
            base_lr=learning_rate,
            max_lr=1e-3,
            step_size= 8 * cyclic_lr_step_size)) # recommended coeficient is from 2 to 8

    return callbacks

def get_model(args, params):
    models_collection = {
        'inception': lambda p: get_inception_v3(
            **p,
            train_ds = train_ds,
            train_steps = train_steps,
            weights = weights,
            freeze_layers_number = 172,#249
            ),
        'vgg': lambda p: get_vgg_model(**p),
        'alex': lambda p: get_alex_model(**p),
        'all_cnn': lambda p: get_all_cnn_model(**p),
        'inception_clean': lambda p: get_inception_v3_clean(**p),
        'efficient': lambda p: get_efficient(**p,
            train_ds = train_ds,
            train_steps = train_steps,
            weights = weights,
            freeze_layers_number = 620
            )
    }
    model = models_collection[args['model']](params)

    # Load weights
    if args['checkpoint_path']:
        model.load_weights(args['checkpoint_path'])

    return model

# Get args
args = init_env()
params = get_training_params(args)

# Create input objects
augment_lambda = None
if args['augment']:
    augment_lambda = lambda ds: augment(ds, args['img_size'], flip=True, hue=True, saturation=True, 
        brighness=True, contrast=True, zoom=True, rotate=True)

train_ds, train_count, val_ds, val_count, weights = create_datasets(
    dataframe_path=args['dataframe_path'],
    base_image_dir=args['image_dir'],
    quality_dataset_path=args['quality_dataset_path'],
    balance_mode=args['balance_mode'],
    img_size=args['img_size'],
    batch_size=args['batch_size'],
    augment_lambda = augment_lambda)
train_steps = 5000 // args['batch_size']  # train_count // args['batch_size']

# Create model
model = get_model(args, params)

# Train
experiment_dir = create_experiment(params, args)
callbacks = get_callbacks(save_best_models=True,
                          best_models_dir=os.path.join(experiment_dir, 'models'),
                          early_stopping=True,
                          reduce_lr_on_plateau=False,
                          training_history=True,
                          metrics_plot_dir=experiment_dir,
                          cyclic_lr=args['cyclic_lr'],
                          cyclic_lr_step_size=train_steps,
                          learning_rate=args['learning_rate'],
                          class_names=CLASS_NAMES)

try:
    history = model.fit(train_ds, 
                        steps_per_epoch = train_steps,
                        validation_data = val_ds, 
                        validation_steps = val_count // args['batch_size'],
                        class_weight = weights,
                        epochs = 500,
                        callbacks = callbacks
                        )
except Exception as e:
    print('Something happened during train process.', str(e))

# Validate
create_confusion_matrix(model, val_ds, val_count //
                        args['batch_size'], CLASS_NAMES, experiment_dir)
