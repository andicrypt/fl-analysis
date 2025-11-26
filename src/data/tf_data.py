import itertools
import math

import numpy as np
import tensorflow as tf

from src.data import image_augmentation
from src.data import emnist

AUTOTUNE = tf.data.AUTOTUNE


# ======================================================================
# Base Dataset Class (Upgraded & Clean)
# ======================================================================
class Dataset:
    def __init__(self, x_train, y_train, batch_size=50, x_test=None, y_test=None):
        self.batch_size = batch_size

        self.x_train, self.y_train = self.shuffle(x_train, y_train)
        self.x_test, self.y_test = x_test, y_test

        self.x_aux = None
        self.y_aux = None
        self.mal_aux_labels = None
        self.x_aux_test = None
        self.mal_aux_labels_test = None

        # modern dataset
        self.fg = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

    # --------------------------------------------------------------
    def shuffle(self, x, y):
        perm = np.random.permutation(x.shape[0])
        return x[perm], y[perm]

    # --------------------------------------------------------------
    def get_data(self):
        shuffle_size = min(self.x_train.shape[0], 10000)
        return (
            self.fg
            .shuffle(shuffle_size)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(AUTOTUNE)
        )

    # --------------------------------------------------------------
    def get_aux(self, mal_num_batch):
        """Yield malicious batches (simple numpy slicing)."""
        total_batches = self.x_aux.shape[0] // self.batch_size

        if total_batches < 1:
            yield self.x_aux, self.mal_aux_labels

        for i in range(total_batches):
            bx = self.x_aux[i*self.batch_size:(i+1)*self.batch_size]
            by = self.mal_aux_labels[i*self.batch_size:(i+1)*self.batch_size]
            yield bx, by

    # --------------------------------------------------------------
    def get_data_with_aux(self, insert_aux_times, num_batches,
                          pixel_pattern=None, noise_level=None):
        """
        Mix aux and normal batches (Bagdasaryan-style poisoning).
        """

        if insert_aux_times == 0:
            return (
                self.fg.shuffle(self.x_train.shape[0])
                .batch(self.batch_size, drop_remainder=True)
                .prefetch(AUTOTUNE)
            )

        multiplier = max(
            float(insert_aux_times) / float(self.mal_aux_labels.shape[0]),
            1
        )
        number_of_mal_items = int(multiplier * num_batches)

        r1 = insert_aux_times
        r2 = self.batch_size - insert_aux_times

        # normal ds
        normal_mult = max(
            float(num_batches) * float(self.batch_size) / self.x_train.shape[0],
            1
        )
        normal_fg = (
            self.fg
            .repeat(int(math.ceil(normal_mult)))
            .shuffle(self.x_train.shape[0])
            .batch(r2, drop_remainder=True)
        )

        # malicious ds
        mal_fb = (
            tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels))
            .repeat(number_of_mal_items)
            .shuffle(number_of_mal_items)
        )

        if noise_level is not None:
            mal_fb = mal_fb.map(image_augmentation.add_noise(noise_level), num_parallel_calls=AUTOTUNE)

        mal_fb = (
            mal_fb
            .batch(r1, drop_remainder=True)
        )

        # zip malicious and clean
        zipped = tf.data.Dataset.zip((mal_fb, normal_fg)).map(
            lambda x, y: (
                tf.concat([x[0], y[0]], axis=0),
                tf.concat([x[1], y[1]], axis=0)
            )
        )

        result = zipped.unbatch()
        return result.batch(self.batch_size, drop_remainder=True).take(num_batches)

    # --------------------------------------------------------------
    def get_aux_test_generator(self, aux_size):
        if aux_size == 0:
            return (
                tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
                .batch(self.batch_size)
                .prefetch(AUTOTUNE)
            )

        return (
            tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
            .repeat(aux_size)
            .batch(self.batch_size)
            .prefetch(AUTOTUNE)
        )

    # ==================================================================
    # Utility functions
    # ==================================================================
    @staticmethod
    def keep_samples(x_train, y_train, number_of_samples):
        if number_of_samples == -1:
            return x_train, y_train
        perm = np.random.permutation(number_of_samples)
        return x_train[perm], y_train[perm]

    @staticmethod
    def keep_samples_iterative(x_train, y_train, number_of_samples):
        if number_of_samples == -1:
            return x_train, y_train

        perms = [
            np.random.permutation(min(number_of_samples, x.shape[0]))
            for x in x_train
        ]
        return (
            [x[p, :] for x, p in zip(x_train, perms)],
            [y[p] for y, p in zip(y_train, perms)],
        )

    @staticmethod
    def apply_trigger(x_aux):
        triggersize = 4
        trigger = np.ones((x_aux.shape[0], triggersize, triggersize, 1))
        out = x_aux.copy()
        out[:, :triggersize, :triggersize, :] = trigger
        return out

    # ==================================================================
    # Dataset loaders
    # ==================================================================
    @staticmethod
    def get_mnist_dataset(number_of_samples):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train, x_test = x_train/255.0, x_test/255.0
        x_train = x_train[..., np.newaxis].astype(np.float32)
        x_test = x_test[..., np.newaxis].astype(np.float32)

        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_fmnist_dataset(number_of_samples):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

        x_train, x_test = x_train/255.0, x_test/255.0
        x_train = x_train[..., np.newaxis].astype(np.float32)
        x_test = x_test[..., np.newaxis].astype(np.float32)

        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_cifar10_dataset(number_of_samples):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)

        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)

        # normalize by training statistics
        mean = np.mean(x_train, axis=0)
        x_train -= mean
        x_test -= mean
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_emnist_dataset(number_of_samples, number_of_clients, normalize_mnist_data):
        train_dataset, test_dataset = emnist.load_data()

        x_train = np.array([1.0 - np.array(v['pixels']) for v in train_dataset])
        y_train = np.array([v['label'] for v in train_dataset], dtype=np.uint8)

        x_test = np.array([1.0 - np.array(v['pixels']) for v in test_dataset])
        y_test = np.array([v['label'] for v in test_dataset], dtype=np.uint8)

        if normalize_mnist_data:
            mean, std = 0.036910772, 0.16115953
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

        # client batching
        if number_of_clients < x_train.shape[0]:
            assigns = np.random.randint(0, number_of_clients, x_train.shape[0])

            new_train_x, new_train_y = [], []
            new_test_x, new_test_y = [], []

            for cid in range(number_of_clients):
                new_train_x.append(np.concatenate(x_train[assigns == cid], axis=0))
                new_train_y.append(np.concatenate(y_train[assigns == cid], axis=0))
                new_test_x.append(np.concatenate(x_test[assigns == cid], axis=0))
                new_test_y.append(np.concatenate(y_test[assigns == cid], axis=0))

            if number_of_samples == -1:
                per_client = -1
            else:
                per_client = number_of_samples // number_of_clients

            x_train, y_train = Dataset.keep_samples_iterative(new_train_x, new_train_y, per_client)
            x_test, y_test = Dataset.keep_samples_iterative(new_test_x, new_test_y, min(per_client, 500))

        x_train = [x.astype(np.float32)[..., np.newaxis] for x in x_train]
        x_test = [x.astype(np.float32)[..., np.newaxis] for x in x_test]

        return (x_train, y_train), (x_test, y_test)


# ======================================================================
# ImageGeneratorDataset (Upgraded)
# ======================================================================
class ImageGeneratorDataset(Dataset):
    def get_aux_test_generator(self, aux_size):
        if aux_size == 0:
            return (
                tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
                .batch(self.batch_size)
                .prefetch(AUTOTUNE)
            )
        return (
            tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
            .repeat(aux_size)
            .batch(self.batch_size)
            .map(image_augmentation.test_aux_augment, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )

    def get_data(self):
        return (
            self.fg
            .shuffle(self.x_train.shape[0])
            .batch(self.batch_size)
            .map(image_augmentation.augment, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )

    def get_aux(self, mal_num_batch):
        multiplier = max(float(self.batch_size) / self.mal_aux_labels.shape[0], 1)
        num_items = int(multiplier * mal_num_batch)

        return (
            tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels))
            .repeat(num_items)
            .batch(self.batch_size)
            .map(image_augmentation.train_aux_augment, num_parallel_calls=AUTOTUNE)
        )


# ======================================================================
# PixelPattern Dataset
# ======================================================================
class PixelPatternDataset(ImageGeneratorDataset):

    def __init__(self, x_train, y_train, target_label, batch_size=50, x_test=None, y_test=None):
        super().__init__(x_train, y_train, batch_size, x_test, y_test)

        self.x_aux, self.y_aux = self.x_train, self.y_train
        self.mal_aux_labels = np.repeat(target_label, self.y_aux.shape).astype(np.uint8)
        self.pixel_pattern = "basic"

    def get_aux_test_generator(self, aux_size):
        pixel_cb = image_augmentation.add_pixel_pattern(self.pixel_pattern)

        if aux_size == 0:
            return (
                tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
                .map(pixel_cb)
                .batch(self.batch_size)
                .prefetch(AUTOTUNE)
            )

        return (
            tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test))
            .repeat(aux_size)
            .batch(self.batch_size)
            .map(image_augmentation.test_aux_augment, num_parallel_calls=AUTOTUNE)
            .map(pixel_cb)
            .prefetch(AUTOTUNE)
        )


# ======================================================================
# GeneratorDataset (unchanged except AUTOTUNE fix)
# ======================================================================
class GeneratorDataset(Dataset):
    def __init__(self, generator, batch_size):
        super().__init__([], [], 0)
        self.generator = generator
        self.batch_size = batch_size

    def get_data(self):
        return (
            self.generator
            .batch(self.batch_size)
            .prefetch(AUTOTUNE)
        )
