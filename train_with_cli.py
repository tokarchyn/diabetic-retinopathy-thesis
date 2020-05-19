from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets, layers, models

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
IN_COLAB = 'google.colab' in sys.modules

from modules.cyclic_lr import CyclicLR
from modules.f_score import F1Score
from modules.plots import plot_confusion_matrix
from modules.augmenter import augment
from modules.training_history_callback import TrainingHistoryCallback
from models.alex_model import get_alex_model
from models.all_cnn_model import get_all_cnn_model
from models.inception_v3_model import get_inception_v3
from models.vgg_16_model import get_vgg_model

pd.options.mode.chained_assignment = None
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %precision % .5f
np.set_printoptions(suppress=True, precision=5)

# Define constants
CLASS_NAMES = np.array(
    ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
CLASS_INDEXES = [0, 1, 2, 3, 4]


def fetch_data_from_gdrive(project_dir, remote_project_dir, zip_name):
    target_zip = os.path.join(project_dir, zip_name)
    # !mkdir - p "{project_dir}"
    # !cp "{remote_project_dir}/{zip_name}" "{target_zip}"
    # !unzip - q "{target_zip}"
    # !rm "{target_zip}"


def init_env():
    args = {}
    if IN_COLAB:
        args['remote_project_dir'] = 'drive/My Drive/diabetic-retinopathy-thesis'
        args['project_dir'] = '/content'
        args['image_dir'] = os.path.join(
            args['project_dir'], 'train_processed')
        args['dataframe_path'] = os.path.join(
            args['remote_project_dir'], 'trainLabels_full.csv')
        args['experiments_dir'] = os.path.join(
            args['remote_project_dir'], 'experiments')
        args['quality_dataset_path'] = None
        args['model'] = 'inception'
        args['cyclic_lr'] = True
        args['balance_mode'] = None
        args['augment'] = True

        from google.colab import drive
        drive.mount('/content/drive')

        fetch_data_from_gdrive(args['project_dir'],
                               args['remote_project_dir'],
                               'train_processed.zip')
    else:
        old_argv = sys.argv
        if sys.argv[-1].endswith('json'):
            sys.argv = ['']
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
                            default='inception')
        parser.add_argument('--cyclic_lr', action='store_true')
        parser.add_argument('--augment', action='store_true')
        parser.add_argument('--balance_mode', type=str,
                            default=None)
        parsed_args = parser.parse_args()
        args = vars(parsed_args)
        sys.argv = old_argv

    if args['gpu_id'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args['gpu_id']
        tf_device='/gpu:0'

    args['img_size'] = 512
    args['batch_size'] = 16
    args['learning_rate'] = 0.00005
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


def prepare(ds, shuffle_buffer_size=200):
    # scale pixels between -1 and 1
    # ds = ds.map(lambda img, level: (tf.keras.applications.inception_v3.preprocess_input(img), level),
    #             num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def dataset_from_tensor_slices(df):
    return tf.data.Dataset.from_tensor_slices((
        df['image_path'].to_numpy(copy=True),
        df['level'].to_numpy(copy=True)))


def create_datasets(train_df, val_df, img_size, batch_size, augment_lambda=None):
    train_ds = dataset_from_tensor_slices(train_df)
    val_ds = dataset_from_tensor_slices(val_df)

    def process_path_local(file_path, level): return process_path(
        file_path, level, img_size)

    train_ds = train_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    if augment_lambda is not None:
        print('Add data augmentation')
        train_ds = augment_lambda(train_ds)
    train_ds = prepare(train_ds)
    train_ds = train_ds.batch(batch_size)

    val_ds = val_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    val_ds = prepare(val_ds)
    val_ds = val_ds.batch(batch_size)

    return train_ds, len(train_df), val_ds, len(val_df)


# Callbacks and metrics
# Helps to persist all metrics and weights in situation when you interupt training


def get_callbacks(save_best_models=True, best_models_dir=None,
                  early_stopping=True,
                  reduce_lr_on_plateau=False,
                  training_history=True, metrics_plot_dir=None,
                  cyclic_lr=True, cyclic_lr_step_size=-1,
                  class_names=[]):
    callbacks = []
    if save_best_models:
        Path(best_models_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                best_models_dir, 'e_{epoch:02d}-acc_{val_accuracy:.2f}-f1_{val_f1_score:.2f}.hdf5'),
            monitor='val_f1_score',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto'))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            patience=30,
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
            ['loss', 'accuracy', 'f1_score', 'lr'],
            metrics_plot_dir = metrics_plot_dir,
            save_weights = False,
            class_names = class_names))
    if cyclic_lr:
        callbacks.append(CyclicLR(
            mode='triangular',
            base_lr=1e-3,
            max_lr=1e-6,
            step_size= 4 * cyclic_lr_step_size)) # recommended coeficient is from 2 to 8

    return callbacks


def top_2_accuracy(in_gt, in_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(in_gt, in_pred, k=2)


def get_metrics():
    return ['accuracy', F1Score(len(CLASS_INDEXES)), top_2_accuracy]


# Train and validation


def train(model, train_ds, train_steps, val_ds, val_steps, epochs, callbacks, weights=None):
    print('Start training.')
    history = model.fit(train_ds, steps_per_epoch=train_steps,
                        validation_data=val_ds, validation_steps=val_steps,
                        class_weight=weights,
                        epochs=epochs,
                        callbacks=callbacks
                        )
    print('Training finished.')
    return history


def create_confusion_matrix(model, dataset, steps, target_names, save_dest=None):
    print('Creating confusion matrix.')
    it = iter(dataset)
    true_labels_glob = []
    pred_labels_glob = []

    for i in range(0, steps):
        image_batch, true_labels = next(it)
        true_labels_glob.extend(np.argmax(true_labels, axis=1))
        pred = model.predict(image_batch)
        pred_labels_glob.extend(np.argmax(pred, axis=1))

    plot_confusion_matrix(
        true_labels_glob, pred_labels_glob, target_names, save_dest)
    print('Confusion matrix was saved to', save_dest)


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
    'metrics': get_metrics(),
    'lr': args['learning_rate'],
    'activation': tf.keras.layers.LeakyReLU(alpha=0.3)
}
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
    'all_cnn': lambda p: get_all_cnn_model(**p)
}
model = models_collection[args['model']](params)
# model.summary()

# Train
experiment_dir = create_experiment(params, args)
callbacks = get_callbacks(save_best_models=True,
                          best_models_dir=os.path.join(experiment_dir, 'models'),
                          early_stopping=False,
                          reduce_lr_on_plateau=False,
                          training_history=True,
                          metrics_plot_dir=experiment_dir,
                          cyclic_lr=args['cyclic_lr'],
                          cyclic_lr_step_size=train_steps,
                          class_names=CLASS_NAMES)
history = train(model=model,
                train_ds=train_ds,
                train_steps=train_steps,
                val_ds=val_ds,
                val_steps=val_count // args['batch_size'],
                epochs=500,
                weights=weights,
                callbacks=callbacks)

# Validate
create_confusion_matrix(model, val_ds, val_count //
                        args['batch_size'], CLASS_NAMES, experiment_dir)
