from sklearn.model_selection import train_test_split
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
from pathlib import Path
import glob
import os
import pandas as pd
import json

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
    select = df['image_path'].apply(lambda p: Path(p).stem not in quality_dict or quality_dict[Path(p).stem] != 2)
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

def decode_img(img, img_size):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

CLASS_INDEXES = [0, 1, 2, 3, 4]
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


def create_datasets_from_dataframes(train_df, val_df, img_size, batch_size, augment_lambda=None, shuffle_buffer_size=1000):
    train_ds = dataset_from_tensor_slices(train_df)
    val_ds = dataset_from_tensor_slices(val_df)

    process_path_local = lambda file_path, level: process_path(file_path, level, img_size)

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

def create_datasets(dataframe_path, base_image_dir, quality_dataset_path=None, 
                    balance_mode = None, img_size=512, batch_size=16, 
                    augment_lambda=None, shuffle_buffer_size=1000):

    train_df, val_df, weights = prepare_data(
        dataframe_path, base_image_dir, quality_dataset_path, balance_mode)

    train_ds, train_count, val_ds, val_count = create_datasets_from_dataframes(
        train_df, val_df, img_size, batch_size, augment_lambda, shuffle_buffer_size)

    return train_ds, train_count, val_ds, val_count, weights