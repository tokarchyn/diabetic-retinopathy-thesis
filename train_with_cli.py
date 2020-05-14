from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import json
import argparse
import itertools
from pathlib import Path
import datetime
import seaborn as sns
from IPython.display import display, clear_output
import glob
import copy
import os
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import sys
IN_COLAB = 'google.colab' in sys.modules

plt.ioff()

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
            args['remote_project_dir'], 'trainLabels.csv')
        args['experiments_dir'] = os.path.join(
            args['remote_project_dir'], 'experiments')
        args['quality_dataset_path'] = 'image_quality.csv'

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
                            default='trainLabels.csv')
        parser.add_argument('--quality_dataset_path',
                            type=str, default='image_quality.csv')
        parser.add_argument('--experiments_dir', type=str,
                            default='experiments')
        parsed_args = parser.parse_args()
        args = vars(parsed_args)
        sys.argv = old_argv

    args['img_size'] = 512
    args['batch_size'] = 32
    print('Arguments:', json.dumps(args))
    return args

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
    select = df['image_path'].apply(lambda p: Path(
        p).stem not in quality_dict or quality_dict[Path(p).stem] == 0)
    print('Images to exclude:', len(select[select == False]))
    df = df.loc[select]
    return df


def prepare_data(dataframe_path, base_image_dir, quality_dataset_path):
    if not os.path.exists(base_image_dir):
        raise NameError('Base image path doesnt exist', base_image_dir)
    df = load_df(dataframe_path, base_image_dir)
    df = remove_unexist(df, base_image_dir)
    df = shuffle(df)
    # df = shrink_dataset(df, 1000)

    train_df, val_df = train_val_split(df)
    train_df = exclude_by_quality(train_df, quality_dataset_path)
    # train_df = balance_with_mode(train_df, mode='max') # take the same number of samples as majority category has
    # take some samples from each category
    train_df = balance(train_df, counts={
                       0: 6000, 1: 6000, 2: 6000, 3: 6000, 4: 6000})
    # train_df = balance_with_mode(train_df, mode='min') # take the same number of samples as minority category has
    train_df = shuffle(train_df)
    weights = calc_weights(train_df)

    return train_df, val_df, weights

# Augmentation functions


def rotate(x):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def flip(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def color(x):
    x = tf.image.random_hue(x, 0.04)
    x = tf.image.random_saturation(x, 0.9, 1.1)
    x = tf.image.random_brightness(x, 0.04)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x


def zoom(x, img_size):
    # Generate 20 crop settings, ranging from a 1% to 10% crop.
    scales = list(np.arange(0.9, 1, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes,
                                         box_indices=np.zeros(len(scales)),
                                         crop_size=(img_size, img_size))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(
        shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


def augment(dataset, img_size, aug_probability=1):
    def zoom_local(x): return zoom(x, img_size)
    augmentations = [flip, rotate, color, zoom_local]

    def augment_map(img, level, aug_fun):
        return (aug_fun(img), level)
        # return (tf.cond(tf.math.argmax(level, axis = 0) == 0, lambda: img, lambda: aug_fun(img)), level)
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

# Create Tensorflow's dataset


def get_input_shape(img_size):
    return (img_size, img_size, 3)


def decode_img(img, img_size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_size, img_size])


def get_label(level):
    return tf.cast(level == CLASS_INDEXES, dtype=tf.float32)


def process_path(file_path, level, img_size):
    label = get_label(level)
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, label


def prepare(ds, shuffle_buffer_size=200):
    ds = ds.map(lambda img, level: (tf.image.per_image_standardization(img), level),
                num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def dataset_from_tensor_slices(df):
    return tf.data.Dataset.from_tensor_slices((
        df['image_path'].to_numpy(copy=True),
        df['level'].to_numpy(copy=True)))


def create_datasets(train_df, val_df, img_size, batch_size):
    train_ds = dataset_from_tensor_slices(train_df)
    val_ds = dataset_from_tensor_slices(val_df)

    def process_path_local(file_path, level): return process_path(
        file_path, level, img_size)

    train_ds = train_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    train_ds = augment(train_ds, img_size)
    train_ds = prepare(train_ds)
    train_ds = train_ds.batch(batch_size)

    val_ds = val_ds.map(process_path_local, num_parallel_calls=AUTOTUNE)
    val_ds = prepare(val_ds)
    val_ds = val_ds.batch(batch_size)

    return train_ds, len(train_df), val_ds, len(val_df)

# Visualisation


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


def plot_metric(metrics, metric_name, save_dest=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(metrics[metric_name])
    plt.plot(metrics['val_' + metric_name])
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    if save_dest:
        plt.savefig(os.path.join(save_dest, metric_name + '.png'))
    else:
        plt.show()
    plt.close(fig)


def plot_f1(metrics, save_dest=None):
    fig, m_axs = plt.subplots(2, 1, figsize=(12, 16))

    m_axs[0].plot(metrics['f1_score'])
    m_axs[0].set_ylabel('f1_score')
    m_axs[0].set_xlabel('epoch')
    m_axs[0].set_title('Training')
    m_axs[0].legend(CLASS_NAMES, loc='upper left')

    m_axs[1].plot(metrics['val_f1_score'])
    m_axs[1].set_ylabel('val_f1_score')
    m_axs[1].set_xlabel('epoch')
    m_axs[1].set_title('Validation')
    m_axs[1].legend(CLASS_NAMES, loc='upper left')

    if save_dest:
        fig.savefig(os.path.join(save_dest, 'f1_score.png'))
    else:
        plt.show()
    plt.close(fig)

    plot_metric({
        "f1_score_average": np.array(metrics['f1_score']).mean(axis=1),
        "val_f1_score_average": np.array(metrics['val_f1_score']).mean(axis=1)},
        'f1_score_average',
        save_dest)


def plot_confusion_matrix(true_lables, pred_labels, target_names, save_dest=None):
    cm = confusion_matrix(true_lables, pred_labels)
    cmap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_dest:
        fig.savefig(os.path.join(save_dest, 'confusion_matrix.png'))
    else:
        plt.show()
    plt.close(fig)

# Callbacks and metrics
# Helps to persist all metrics and weights in situation when you interupt training


class TrainingHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, metrics, save_weights=True, metrics_plot_dir=None):
        self.metrics = {}
        self.weights = []
        self.save_weights = save_weights
        self.metrics_plot_dir = metrics_plot_dir
        for m in metrics:
            self.metrics[m] = []
            self.metrics['val_' + m] = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key in self.metrics.keys():
                self.metrics[key].append(logs.get(key))

        if self.metrics_plot_dir:
            for key in self.metrics.keys():
                if not key.startswith('val_'):
                    if key == 'f1_score':
                        plot_f1(self.metrics, self.metrics_plot_dir)
                    else:
                        plot_metric(self.metrics, key, self.metrics_plot_dir)

        if self.save_weights:
            self.weights.append([])
            for l in model.layers:
                self.weights[len(self.weights)-1].append(l.get_weights())


def get_callbacks(save_best_models=True, best_models_dir=None,
                  early_stopping=True,
                  reduce_lr_on_plateau=True,
                  training_history=True, metrics_plot_dir=None):
    callbacks = []
    if save_best_models:
        Path(best_models_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                best_models_dir, 'e_{epoch:02d}-acc_{val_accuracy:.2f}-f1_{val_f1_score}.hdf5'),
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
            patience=10,
            verbose=1,
            mode='auto',
            epsilon=0.0001,
            cooldown=5,
            min_lr=0.00001))
    if training_history:
        Path(metrics_plot_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(TrainingHistoryCallback(
            ['loss', 'accuracy', 'f1_score'],
            metrics_plot_dir=metrics_plot_dir))

    return callbacks


def top_2_accuracy(in_gt, in_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(in_gt, in_pred, k=2)


class FBetaScore(tf.keras.metrics.Metric):
    """Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.
    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(
        self,
        num_classes,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
        **kwargs
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, micro, macro, weighted]"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError(
                    "The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    # TODO: Add sample_weight support, currently it is
    # ignored during calculations.
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(
                y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
        self.weights_intermediate.assign_add(_count_non_zero(y_true))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(
                    self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_states(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class F1Score(FBetaScore):
    """Computes F-1 Score.
    It is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class
    and multi-label classification.
    F-1 = 2 * (precision * recall) / (precision + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-1 Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(
        self,
        num_classes,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=None,
        **kwargs
    ):
        super().__init__(num_classes, average, 1.0, threshold, name=name, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        del base_config["beta"]
        return base_config


def get_metrics():
    return ['accuracy', F1Score(len(CLASS_INDEXES)), top_2_accuracy]

# Models


def get_alex_model(input_shape):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=4, activation="relu",  padding="same",
                     input_shape=input_shape))
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=get_metrics())
    return model


def get_custom_model(input_shape):
    model = models.Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASS_NAMES), activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=get_metrics())
    return model


def get_vgg_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu",
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00002)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=get_metrics())
    return model


def get_inception_v3(train_ds, train_steps, weights, freeze_layers_number, input_shape):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)
    x = base_model.output
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001), loss='categorical_crossentropy')

    # train the model on the new data for a few epochs
    model.fit(train_ds, steps_per_epoch=train_steps,
              epochs=3, class_weight=weights)

    for layer in model.layers[:freeze_layers_number]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_number:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=get_metrics())
    return model


def get_all_cnn_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides=1, activation="relu",  padding="valid",
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), strides=1, activation="relu",  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), strides=2, activation="relu",  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides=1, activation="relu",  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=2, activation="relu",  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=1,
                     activation="relu",  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=2,
                     activation="relu",  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides=1,
                     activation="relu",  padding="valid"))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (1, 1), strides=1,
                     activation="relu",  padding="valid"))
    model.add(BatchNormalization())
    model.add(Conv2D(5, (1, 1), strides=1, activation="relu",  padding="valid"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(len(CLASS_NAMES), activation="softmax"))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy", metrics=get_metrics())
    return model

# Train and validation


def train(model, train_ds, train_steps, val_ds, val_steps,
          experiment_dir, epochs, weights=None):
    print('Start training.')
    history = model.fit(train_ds, steps_per_epoch=train_steps,
                        validation_data=val_ds, validation_steps=val_steps,
                        class_weight=weights,
                        epochs=epochs,
                        callbacks=get_callbacks(
                            save_best_models=False,
                            best_models_dir=os.path.join(
                                experiment_dir, 'models'),
                            early_stopping=False,
                            reduce_lr_on_plateau=True,
                            training_history=True,
                            metrics_plot_dir=experiment_dir)
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
    args['dataframe_path'], args['image_dir'], args['quality_dataset_path'])
train_ds, train_count, val_ds, val_count = create_datasets(
    train_df=train_df,
    val_df=val_df,
    img_size=args['img_size'],
    batch_size=args['batch_size'])
train_steps = 5000 // args['batch_size']  # train_count // args['batch_size']

# Create model
input_shape = get_input_shape(args['img_size'])
try:
    del model
except:
    print('There is no model defined')
# model = get_model(input_shape)
# model = get_vgg_model(input_shape)
# model = get_alex_model(input_shape)
model = get_inception_v3(
    train_ds=train_ds,
    train_steps=train_steps,
    weights=weights,
    freeze_layers_number=172,
    input_shape=input_shape)
# model = get_all_cnn_model(input_shape)
# model.summary()

# Train
experiment_dir = os.path.join(args['experiments_dir'],
                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
history = train(model=model,
                train_ds=train_ds,
                train_steps=train_steps,
                val_ds=val_ds,
                val_steps=val_count // args['batch_size'],
                experiment_dir=experiment_dir,
                epochs=500,
                weights=weights)

# Validate
create_confusion_matrix(model, val_ds, val_count //
                        args['batch_size'], CLASS_NAMES, experiment_dir)
