# Imports
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
import sys

import math
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import copy
import glob
import datetime

import tensorflow as tf

pd.options.mode.chained_assignment = None
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Define constants
CLASS_NAMES = np.array(
    ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'])
CLASS_INDEXES = [0, 1, 2, 3, 4]
WIDTH = HEIGHT = 512
BATCH_SIZE = 32

if len(sys.argv) < 2:
    print('You need to specify the path to images')
    sys.exit()

BASE_IMAGE_DIR = sys.argv[1]
TRAIN_LABELS_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'trainLabels.csv')
LOGS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs')


# Define functions
def load_df():
    df = pd.read_csv(TRAIN_LABELS_PATH)
    df['image_path'] = df['image'].map(lambda p: os.path.normpath(
        os.path.join(BASE_IMAGE_DIR, p + '.jpeg')))
    df = df.drop(columns=['image'])
    print('Number of images in dataframe is', len(df))
    return df


def remove_unexist(df):
    all_images = glob.glob(BASE_IMAGE_DIR + "/*")
    all_images = [os.path.normpath(p) for p in all_images]
    while len(all_images) == 0:
        all_images = glob.glob(BASE_IMAGE_DIR + "/*")
    print('Found', len(all_images), 'images')
    df['exists'] = df['image_path'].map(lambda p: p in all_images)
    print('First in all images', all_images[0])
    print('First in dataframe', df['image_path'][0])
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
    print('Weights for each level:\n', weights)
    return weights


def get_samples_of_level(df, level, count):
    to_add = []
    it = iter(df[df['level'] == level].iterrows())

    while count > len(to_add):
        _, row = None, None
        try:
            _, row = next(it)
        except StopIteration:
            it = iter(df[df['level'] == level].iterrows())
            _, row = next(it)
        to_add.append(copy.deepcopy(row))

    return to_add

# Upsampling and Downsampling based on multiplicator or exact count for each category
# For example multipliers={0: 2} will take 2 times more samples of category 0 than exist.


def balancing(df, multipliers=None, counts=None):
    max_level_count = df['level'].value_counts().max()

    for level in df['level'].unique():
        count_of_level = df[df['level'] == level].count()[0]

        count_diff = 0
        if multipliers != None:
            if level in multipliers:
                count_diff = int(count_of_level *
                                 multipliers[level]) - count_of_level
        elif counts != None:
            if level in counts:
                count_diff = counts[level] - count_of_level
        else:
            if count_of_level == max_level_count:
                continue
            count_diff = max_level_count - count_of_level

        if count_diff == 0:
            continue
        print('Need to add(or remove)', count_diff, 'copies of level',
              level, 'where count of level is', count_of_level)
        if count_diff < 0:
            df_level = df[df['level'] == level]
            df = df.drop(df_level.sample(count_diff * -1).index)
        else:
            df = df.append(get_samples_of_level(
                df, level, count_diff), ignore_index=True)

    return df


def shrink_dataset_equally(df, number_of_each_level=None):
    levels = df['level'].unique()

    if number_of_each_level is None:
        number_of_each_level = df['level'].value_counts().min()

    def get_rows(df_tmp):
        size = len(df_tmp)
        return df_tmp.sample(number_of_each_level) if size >= number_of_each_level else df_tmp

    df_tmp = get_rows(df[df['level'] == levels[0]])
    for l in levels[1:]:
        df_tmp = df_tmp.append(
            get_rows(df[df['level'] == l]), ignore_index=True)
    return df_tmp


def shrink_dataset(df, count):
    return df[:count]


def shuffle(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def prepare_data():
    df = load_df()
    df = remove_unexist(df)
    df = shuffle(df)
    # df = shrink_dataset(df, 10000)

    train_df, val_df = train_val_split(df)
    # train_df = balancing(train_df) # take the same number of samples as majority category has
    # train_df = balancing(train_df, multipliers={0:0.3, 1:0.7, 2:0.4, 3:2, 4:2})
    # take some samples from each category
    train_df = balancing(
        train_df, counts={0: 6000, 1: 6000, 2: 6000, 3: 6000, 4: 6000})
    # train_df = shrink_dataset_equally(train_df)
    train_df = shuffle(train_df)
    weights = calc_weights(train_df)

    return train_df, val_df, weights


def rotate(x):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def flip(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def color(x):
    x = tf.image.random_hue(x, 0.06)
    x = tf.image.random_saturation(x, 0.8, 1.2)
    x = tf.image.random_brightness(x, 0.04)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    return x


def zoom(x):
    scales = list(np.arange(0.8, 1, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(
            len(scales)), crop_size=(HEIGHT, WIDTH))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(
        shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


def augment(dataset, aug_probability=1):
    augmentations = [flip, rotate]

    def augment_map(img, level, aug_fun):
        # return (aug_fun(img), level)
        return (tf.cond(tf.math.argmax(level, axis=0) == 0, lambda: img, lambda: aug_fun(img)), level)
        # choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        # return (tf.cond(choice < aug_probability, lambda: img, lambda: aug_fun(img)),
        #         level)

    # Add the augmentations to the dataset
    for f in augmentations:
        dataset = dataset.map(lambda img, level: augment_map(
            img, level, f), num_parallel_calls=AUTOTUNE)

    # Make sure that the values are still in [0, 1]
    dataset = dataset.map(lambda img, level: (
        tf.clip_by_value(img, 0, 1), level), num_parallel_calls=AUTOTUNE)
    return dataset


def get_input_shape():
    return (HEIGHT, WIDTH, 3)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [WIDTH, HEIGHT])


def get_input_shape():
    return (HEIGHT, WIDTH, 3)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [WIDTH, HEIGHT])


def get_label(level):
    return tf.cast(level == CLASS_INDEXES, dtype=tf.float32)


def process_path(file_path, level):
    label = get_label(level)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare(ds, shuffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def show_batch(image_batch, label_batch, number_to_show=4, predicted_labels=None):
    row_count = math.ceil(number_to_show / 4)
    fig, m_axs = plt.subplots(row_count, 4, figsize=(16, row_count * 4))
    for i, (c_x, c_y, c_ax) in enumerate(zip(image_batch, label_batch, m_axs.flatten())):
        c_ax.imshow(c_x)
        real_level = CLASS_NAMES[c_y == 1][0]
        pred_level = ''
        title = 'Real level: ' + real_level
        if predicted_labels is not None:
            pred_level = CLASS_NAMES[predicted_labels[i]]
            title = title + '\nPredicted one: ' + pred_level
        c_ax.set_title(title, color='g' if pred_level ==
                       '' or real_level == pred_level else 'r')
        c_ax.axis('off')


def dataset_from_tensor_slices(df):
    return tf.data.Dataset.from_tensor_slices((
        df['image_path'].to_numpy(copy=True),
        df['level'].to_numpy(copy=True)))


def create_datasets(train_df, val_df, batch_size=BATCH_SIZE):
    train_ds = dataset_from_tensor_slices(train_df)
    val_ds = dataset_from_tensor_slices(val_df)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = augment(train_ds)
    train_ds = prepare(train_ds)
    train_ds = train_ds.batch(batch_size)

    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = prepare(val_ds)
    val_ds = val_ds.batch(batch_size)

    return train_ds, len(train_df), val_ds, len(val_df)


class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics, save_weights=True):
        self.metrics = {}
        self.weights = []
        self.save_weights = save_weights
        for m in metrics:
            self.metrics[m] = []
            self.metrics['val_' + m] = []

    def on_epoch_end(self, epoch, logs=None):
        for key in self.metrics.keys():
            self.metrics[key].append(logs.get(key))
        if self.save_weights:
            self.weights.append([])
            for l in self.model.layers:
                self.weights[len(self.weights)-1].append(l.get_weights())


def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec, weight_mat,  eps=1e-6, dtype=tf.float32):
    labels = tf.matmul(y_true, col_label_vec)
    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[
                    1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)

    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(
                tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )

    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)

    return tf.math.log(numerator / denominator + eps)


class CohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self,
                 num_classes,
                 name='cohen_kappa_loss',
                 eps=1e-6,
                 dtype=tf.float32):
        super(CohenKappaLoss, self).__init__(
            name=name, reduction=tf.keras.losses.Reduction.NONE)

        self.num_classes = num_classes
        self.eps = eps
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.col_label_vec, [1, num_classes]) -
            tf.tile(self.row_label_vec, [num_classes, 1]),
            2) / tf.cast(tf.pow(num_classes - 1, 2), dtype=dtype)

    def call(self, y_true, y_pred, sample_weight=None):
        return cohen_kappa_loss(
            y_true, y_pred, self.row_label_vec, self.col_label_vec, self.weight_mat, self.eps, self.dtype
        )

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "eps": self.eps,
            "dtype": self.dtype
        }
        base_config = super(CohenKappaLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Recall(tf.keras.metrics.Metric):

    def __init__(self, name='recall_multi', **kwargs):
        super(Recall, self).__init__(name=name, **kwargs)
        self.builtin_metric = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        non_healthy_gt = tf.math.greater(tf.math.argmax(y_true, axis=1), 0)
        non_healthy_pr = tf.math.greater(tf.math.argmax(y_pred, axis=1), 0)
        self.builtin_metric.update_state(non_healthy_gt, non_healthy_pr)

    def result(self):
        return self.builtin_metric.result()

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.builtin_metric.reset_states()


def get_callbacks(save_best_models=True, early_stopping=True, reduce_lr_on_plateau=True, tb_log=True):
    callbacks = []
    if save_best_models:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            'models/model.e_{epoch:02d}-acc_{val_accuracy:.2f}-f1_{val_f1_score}.hdf5',
            monitor='val_accuracy',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto'))
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True))
    if reduce_lr_on_plateau:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=3,
            verbose=1,
            mode='auto',
            epsilon=0.0001,
            cooldown=5,
            min_lr=0.00001))
    if tb_log:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                LOGS_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1))

    return callbacks


def top_2_accuracy(in_gt, in_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(in_gt, in_pred, k=2)


def get_metrics():
    return ['accuracy', tfa.metrics.F1Score(len(CLASS_INDEXES)), Recall(), top_2_accuracy]


def get_model():
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=4, activation="relu",
                     padding="same", input_shape=get_input_shape()))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (11, 11), strides=1,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(Conv2D(384, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=1,
                     padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), strides=1,
                     padding="same",  activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3),  padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=get_metrics())
    return model


# Training
train_df, val_df, weights = prepare_data()
train_ds, train_count, val_ds, val_count = create_datasets(train_df, val_df)
model = get_model()
history = model.fit(train_ds, steps_per_epoch=train_count // BATCH_SIZE,
                    validation_data=val_ds, validation_steps=val_count // BATCH_SIZE,
                    epochs=100,
                    callbacks=get_callbacks(save_best_models=False, early_stopping=False))
