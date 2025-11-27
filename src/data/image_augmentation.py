import tensorflow as tf
import numpy as np
from keras.layers import RandomTranslation
def augment(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.numpy_function(shift, [image], tf.float32)
    image = normalize(image)
    # debug(image, label)

    return image, label

def test_augment(image,label):
    return normalize(image), label

def train_aux_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.numpy_function(shift, [image], tf.float32)
    # image = tf.add(image, tf.random.normal(tf.shape(image), 0, 0.05))
    return image, label

def test_aux_augment(image, label):
    """Augmentation if aux test set is small"""
    return augment(image, label) # same as training

def normalize(image):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # tf.print("Before:", tf.shape(image), tf.math.reduce_std(image))

    # image = tf.image.per_image_standardization(image)
    # image = image - tf.reshape(mean, [1, 1, 1, 3])
    # image = image / tf.reshape(std, [1, 1, 1, 3])

    # tf.print("After:", tf.shape(image), tf.math.reduce_std(image))

    return image

def shift(images):
    shifted = [shift_single(img) for img in images]
    return np.array(shifted, dtype=np.float32)

def shift_single_tf(image, tx, ty):
    """
    image: numpy or tf tensor (H, W, C)
    tx, ty: translation in pixels
    """

    # Convert pixel translation → fraction of image size
    h, w = image.shape[0], image.shape[1]
    tx_frac = tx / h
    ty_frac = ty / w

    # Create deterministic translation layer
    translate = RandomTranslation(
        height_factor=(tx_frac, tx_frac),
        width_factor=(ty_frac, ty_frac),
        fill_mode='nearest'
    )

    # Apply and return numpy array
    out = translate(tf.expand_dims(image, 0))  # add batch dim
    return out[0].numpy()

def shift_single(image):
    h, w = image.shape[0], image.shape[1]

    # Random shift amount
    tx = np.random.uniform(-0.1, 0.1) * h
    ty = np.random.uniform(-0.1, 0.1) * w

    return shift_single_tf(image, tx, ty)


def add_noise_batch(sigma):
    def cb(images, labels):
        images = images + tf.random.normal(tf.shape(images), mean=0, stddev=sigma)
        return images, labels

    return cb


def add_pixel_pattern(pixel_pattern):
    triggersize = 4
    def np_callback(images):
        trigger = np.ones((images.shape[0], triggersize, triggersize, images.shape[-1]))
        images[:, 0:triggersize, 0:triggersize, :] = trigger
        return images

    def cb(images, labels):
        # shape = tf.shape(images)
        # tf.print(shape)
        # print(shape)
        # trigger = tf.ones((shape[0], triggersize, triggersize, shape[-1]))
        # trigger = tf.ones((None, triggersize, triggersize, 3))
        # tf.ones_like
        # d0 = shape[0]
        # tf.print(d0)
        # x = tf.constant(tf.float32, shape=[d0, triggersize, triggersize, 3])
        # trigger = tf.ones_like(x)
        # images[:, 0:triggersize, 0:triggersize, :] = trigger
        # this callback is slower i think
        images = tf.numpy_function(np_callback, [images], tf.float32)

        return images, labels

    return cb


def pixel_pattern_if_needed(needed):
    def no_op(images, labels):
        return images, labels

    if needed:
        return add_pixel_pattern(None)
    else:
        return no_op


def debug(image, label):
    import matplotlib.pyplot as plt

    for i in range(image.shape[0]):
        plt.figure()
        plt.imshow(image[i] + 0.5)
        plt.title(f"Label: {label[i]}")
        plt.show()