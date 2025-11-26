import tensorflow as tf
import numpy as np


# =======================================
# Modern, GPU-accelerated augmenters
# =======================================

AUG_SHIFT = tf.keras.layers.RandomTranslation(
    height_factor=0.1,
    width_factor=0.1,
    fill_mode="nearest"
)

AUG_FLIP = tf.keras.layers.RandomFlip("horizontal")


# =======================================
# Normalization
# =======================================

CIFAR_MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
CIFAR_STD  = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)

def normalize(image):
    # image: [H, W, 3], values in [0,1] or [0,255]
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - CIFAR_MEAN) / CIFAR_STD
    return image


# =======================================
# Main augment functions
# =======================================

def augment(image, label):
    image = AUG_FLIP(image)
    image = AUG_SHIFT(image)
    image = normalize(image)
    return image, label


def test_augment(image, label):
    return normalize(image), label


def train_aux_augment(image, label):
    image = AUG_FLIP(image)
    image = AUG_SHIFT(image)
    return image, label


def test_aux_augment(image, label):
    return augment(image, label)     # identical to training augment


# =======================================
# Pixel noise
# =======================================

def add_noise_batch(sigma):
    def cb(images, labels):
        noise = tf.random.normal(tf.shape(images), mean=0.0, stddev=sigma)
        return images + noise, labels
    return cb


# =======================================
# Pixel Trigger Pattern (Backdoor)
# =======================================

def add_pixel_pattern(size=4):
    """Add a white trigger square at top-left corner."""
    def cb(images, labels):
        B, H, W, C = images.shape
        trigger = tf.ones((B, size, size, C), dtype=images.dtype)

        # Insert trigger (vectorized)
        images = tf.tensor_scatter_nd_update(
            images,
            indices=tf.reshape(
                tf.range(B, dtype=tf.int32), (-1,1)
            ),
            updates=tf.concat([trigger, images[:, size:, :, :]], axis=1)
        )
        return images, labels

    return cb


def pixel_pattern_if_needed(needed):
    return add_pixel_pattern() if needed else lambda x, y: (x, y)


# =======================================
# Optional debug visualization
# =======================================

def debug(image, label):
    import matplotlib.pyplot as plt
    img = image.numpy()
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.astype(np.uint8)

    plt.imshow(img)
    plt.title(f"Label: {label.numpy()}")
    plt.show()
