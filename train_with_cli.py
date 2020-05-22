from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import *

import json
import argparse
from pathlib import Path
import datetime
import glob
import copy
import os
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
plt.ioff()

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
from models.vgg_16_model import get_vgg_model

pd.options.mode.chained_assignment = None
AUTOTUNE = tf.data.experimental.AUTOTUNE

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

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def create_experiment(params, args):
    experiment_dir = os.path.join(args['experiments_dir'],
                                datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    params_to_log = {k: v if is_jsonable(v) else str(v) for k, v in {**params, **args}.items()}
    print(json.dumps(params_to_log), 
        file=open(os.path.join(experiment_dir, 'params.json'), 'w+'))
    return experiment_dir

# Methods to process dataframe


def load_df(dataframe_path, base_image_dir):
    df = pd.read_csv(dataframe_path)
    df['image_path'] = df['image'].astype(str).apply(
        lambda x: os.path.join(base_image_dir, x + '.jpeg'))
    df = df.drop(columns=['image'])
    return df


def remove_unexist(df, base_image_dir):
    all_images = glob.glob(base_image_dir + "/*")
    while len(all_images) == 0:
        all_images = glob.glob(base_image_dir + "/*")
    print('Found', len(all_images), 'images')
    df['exists'] = df['image_path'].map(lambda p: p in all_images)
    df = df[df['exists']].drop(columns=['exists'])
    print('Number of existed images is', len(df))
    return df


def train_val_split(df):
    train_img, val_img = train_test_split(df['image_path'],
                                          test_size=0.20,
                                          random_state=2020,
                                          stratify=df['level'].to_frame())
    train_df = df[df['image_path'].isin(train_img)]
    val_df = df[df['image_path'].isin(val_img)]
    print('Train dataframe size:',
          train_df.shape[0], 'Validation dataframe size:', val_df.shape[0])
    return train_df, val_df


def calc_weights(df):
    level_counts = df['level'].value_counts().sort_index()
    weights = {cls: len(df) / count for cls, count in enumerate(level_counts)}
    if max(weights.values()) - min(weights.values()) < 0.1:
        print('Reset weights because they all are the same:', max(weights.values()))
        weights = None
    else:
        print('Weights for each level:\n', weights)
    return weights


def balance(df, counts):
    new_df = df.iloc[0:0]  # copy only structure
    for level in df['level'].unique():
        df_level = df[df['level'] == level]
        count = len(df_level)
        new_count = counts[level] if level in counts else count
        if count > new_count:
            new_df = new_df.append(
                df_level.drop(df_level.sample(count - new_count).index),
                ignore_index=True)
        elif count < new_count:
            new_df = new_df.append(df_level, ignore_index=True)
            new_df = new_df.append(df_level.sample(
                new_count - count, replace=True), ignore_index=True)
        else:
            new_df = new_df.append(df_level, ignore_index=True)

    print('New counts of dataset\'s categories: ', json.dumps(
        new_df['level'].value_counts().to_dict()))
    return new_df


def balance_with_mode(df, mode='max'):
    counts = df['level'].value_counts()
    new_count = 0
    if mode == 'max':
        new_count = counts.max()
    elif mode == 'min':
        new_count = counts.min()
    new_counts_dict = {level: new_count for level in df['level'].unique()}
    return balance(df, counts=new_counts_dict)


def shuffle(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def exclude_by_quality(df, quality_dataset_path):
    quality_dict = pd.read_csv(quality_dataset_path, index_col='image_name')[
        'quality'].to_dict()
    select = df['image_path'].apply(lambda p: Path(p).stem not in quality_dict or quality_dict[Path(p).stem] == 0)
    print('Images to exclude:', len(select[select == False]))
    df = df.loc[select]
    return df


def prepare_data(dataframe_path, base_image_dir, quality_dataset_path=None, balance_mode = None):
    if not os.path.exists(base_image_dir):
        raise NameError('Base image path doesnt exist', base_image_dir)
    df = load_df(dataframe_path, base_image_dir)
    df = remove_unexist(df, base_image_dir)
    df = shuffle(df)
    # df = shrink_dataset(df, 1000)

    train_df, val_df = train_val_split(df)
    if quality_dataset_path is not None:
        train_df = exclude_by_quality(train_df, quality_dataset_path)
    # take some samples from each category

    if balance_mode == 'max':
        train_df = balance_with_mode(train_df, mode='max') # take the same number of samples as majority category has
    elif balance_mode == 'min':
        train_df = balance_with_mode(train_df, mode='min') # take the same number of samples as minority category has
    elif balance_mode is not None:
        # try parse count
        count = int(balance_mode)
        train_df = balance(train_df, counts={0: count, 1: count, 2: count, 3: count, 4: count})
    train_df = shuffle(train_df)
    weights = calc_weights(train_df)

    return train_df, val_df, weights


# Create Tensorflow's dataset


def get_input_shape(img_size):
    return (img_size, img_size, 3)


def decode_img(img, img_size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [img_size, img_size])
    return img


def get_label(level):
    return tf.cast(level == CLASS_INDEXES, dtype=tf.float32)


def process_path(file_path, level, img_size):
    label = get_label(level)
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, label


def dataset_from_tensor_slices(df):
    return tf.data.Dataset.from_tensor_slices((
        df['image_path'].to_numpy(copy=True),
        df['level'].to_numpy(copy=True)))


def create_datasets(train_df, val_df, img_size, batch_size, augment_lambda=None, shuffle_buffer_size=1000):
    train_ds = dataset_from_tensor_slices(train_df)
    val_ds = dataset_from_tensor_slices(val_df)

    def process_path_local(file_path, level): return process_path(
        file_path, level, img_size)

    train_ds = train_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    if augment_lambda is not None:
        print('Add data augmentation')
        train_ds = augment_lambda(train_ds)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False)
    train_ds = train_ds.batch(batch_size)

    val_ds = val_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size)

    return train_ds, len(train_df), val_ds, len(val_df)


# Callbacks and metrics
# Helps to persist all metrics and weights in situation when you interupt training


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
                best_models_dir, 'e_{epoch:02d}-acc_{val_accuracy:.2f}-ck_{val_cohen_kappa:.2f}.hdf5'),
            monitor='val_cohen_kappa',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='max'))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_cohen_kappa',
            patience=80,
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

# Get args
args = init_env()

# Create input objects
train_df, val_df, weights = prepare_data(
    args['dataframe_path'], args['image_dir'], args['quality_dataset_path'], args['balance_mode'])

augment_lambda = None
if args['augment']:
    augment_lambda = lambda ds: augment(ds, args['img_size'], flip=True, hue=True, saturation=True, 
        brighness=True, contrast=True, zoom=True, rotate=True)

train_ds, train_count, val_ds, val_count = create_datasets(
    train_df=train_df,
    val_df=val_df,
    img_size=args['img_size'],
    batch_size=args['batch_size'],
    augment_lambda = augment_lambda)
train_steps = 5000 // args['batch_size']  # train_count // args['batch_size']

# Create model
params = {
    'input_shape': get_input_shape(args['img_size']),
    'class_number': len(CLASS_NAMES),
    'metrics': ['accuracy', F1Score(len(CLASS_INDEXES)), CohenKappa(len(CLASS_INDEXES))],
    'activation': tf.keras.layers.LeakyReLU(alpha=0.3),
    'kernel_reg': tf.keras.regularizers.l2(5e-4) if args['kernel_reg'] else None,
    'bias_reg': tf.keras.regularizers.l2(5e-4) if args['bias_reg'] else None
}
if args['optimizer'] == 'adam':
    params['optimizer'] = Adam(lr=args['learning_rate'])
elif args['optimizer'] == 'sgd':
    params['optimizer'] = SGD(lr=args['learning_rate'], momentum=args['momentum'])

try:
    del model
except:
    print('There is no model defined')

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
    'inception_clean': lambda p: get_inception_v3_clean(**p)
}
model = models_collection[args['model']](params)
model.summary()

# Load weights
if args['checkpoint_path']:
    model.load_weights(args['checkpoint_path'])

# Train
experiment_dir = create_experiment(params, args)
callbacks = get_callbacks(save_best_models=False,
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
