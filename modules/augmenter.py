import tensorflow as tf
import tensorflow_addons as tfa
import math
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE

def apply_rotate_90(x):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def apply_rotate(x):
    return tfa.image.rotate(x, tf.random.uniform(shape=[], minval=0, maxval=360, dtype=tf.float32) * math.pi / 180)


def apply_flip(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


def apply_zoom(x, img_size):
    # Generate 10 crop settings, ranging from a 1% to 10% crop.
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


def augment(dataset, img_size, flip=True, rotate=True, hue=True, saturation=True, 
    brighness=True, contrast=True, zoom=True):
    augmentations = []
    if flip:
        augmentations.append(apply_flip)
    if rotate:
        augmentations.append(apply_rotate_90)
    if hue:
        augmentations.append(lambda x: tf.image.random_hue(x, 0.1))
    if saturation:
        augmentations.append(lambda x: tf.image.random_saturation(x, 0.9, 1.1))
    if brighness:
        augmentations.append(lambda x: tf.image.random_brightness(x, 0.1))
    if contrast:
        augmentations.append(lambda x: tf.image.random_contrast(x, 0.9, 1.1))
    if zoom:
        augmentations.append(lambda x: apply_zoom(x, img_size))

    def augment_map(img, level, aug_fun):
        return (aug_fun(img), level)

    # Add the augmentations to the dataset
    for f in augmentations:
        dataset = dataset.map(lambda img, level: augment_map(
            img, level, f), num_parallel_calls=AUTOTUNE)

    # Make sure that the values are still in [0, 1]
    dataset = dataset.map(lambda img, level: (
        tf.clip_by_value(img, 0, 1), level), num_parallel_calls=AUTOTUNE)
    return dataset