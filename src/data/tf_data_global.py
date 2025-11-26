from collections import defaultdict
import numpy as np
import tensorflow as tf
import logging

from src.data import image_augmentation


AUTOTUNE = tf.data.AUTOTUNE


class GlobalDataset:
    """
    Represents the full dataset (train + test), and produces:
    - Client datasets (IID or Non-IID)
    - Global test dataset
    - Auxiliary (backdoor) datasets
    """

    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

        self.x_train = []
        self.y_train = []

        self.x_aux_train = []
        self.y_aux_train = []
        self.mal_aux_labels_train = []

        self.x_aux_test = []
        self.y_aux_test = []
        self.mal_aux_labels_test = []

        # Modern TF dataset
        self.test_dataset = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
        )

        self.aux_test_dataset = None

    # ----------------------------------------------------------------------
    # Generic test dataset
    # ----------------------------------------------------------------------

    def get_test_batch(self, batch_size, max_num_batches=-1):
        """Returns a TF dataset for global evaluation."""
        ds = (
            self.test_dataset
            .batch(batch_size)
            .take(max_num_batches)
            .map(image_augmentation.test_augment, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )
        return ds

    # ----------------------------------------------------------------------
    # Backdoor auxiliary dataset builder
    # ----------------------------------------------------------------------

    def build_global_aux(self, mal_clients, num_backdoor_tasks,
                         attack_objective, aux_sample_size, augment_size):

        if np.count_nonzero(mal_clients) == 0:
            return

        assert np.count_nonzero(mal_clients) >= num_backdoor_tasks

        data_x, data_y = self.x_train, self.y_train

        total_x_aux, total_y_aux, total_mal_aux_labels = [], [], []

        if aux_sample_size == -1:
            aux_sample_size = 10_000_000

        total_aux_count = 0
        num_tasks = 0

        for client_id in range(len(data_x)):
            if num_tasks >= num_backdoor_tasks:
                break
            if total_aux_count >= aux_sample_size:
                print("Aux sample limit reached.")
                break

            if mal_clients[client_id]:
                x_me, y_me = data_x[client_id], data_y[client_id]

                # Collect samples of attack class
                inds = np.where(y_me == attack_objective[0])[0]
                logging.debug(f"{client_id}: found {len(inds)} attack samples")

                keep_mask = np.ones(len(x_me), dtype=bool)
                keep_mask[inds] = False

                x_aux = x_me[inds]
                y_aux = y_me[inds]

                # Poison labels
                mal_labels = np.repeat(attack_objective[1], len(y_aux))

                # Respect aux_sample_size
                current = len(y_aux)
                if total_aux_count + current > aux_sample_size:
                    remaining = aux_sample_size - total_aux_count
                    x_aux = x_aux[:remaining]
                    y_aux = y_aux[:remaining]
                    mal_labels = mal_labels[:remaining]
                    current = remaining

                total_x_aux.append(x_aux)
                total_y_aux.append(y_aux)
                total_mal_aux_labels.append(mal_labels)

                # Remove poisoned samples from original client dataset
                data_x[client_id] = x_me[keep_mask]
                data_y[client_id] = y_me[keep_mask]

                assert not np.any(data_y[client_id] == attack_objective[0])
                total_aux_count += current
                num_tasks += 1

        self.x_aux_train = np.concatenate(total_x_aux)
        self.y_aux_train = np.concatenate(total_y_aux)
        self.mal_aux_labels_train = np.concatenate(total_mal_aux_labels).astype(np.uint8)

        # Testing = same as training (your old behavior)
        self.x_aux_test = self.x_aux_train
        self.y_aux_test = self.y_aux_train
        self.mal_aux_labels_test = self.mal_aux_labels_train

        print(f"Generated {len(self.x_aux_train)}/{aux_sample_size} aux samples.")

    # ----------------------------------------------------------------------
    # Auxiliary dataset generator
    # ----------------------------------------------------------------------

    def get_aux_generator(self, batch_size, aux_size,
                          augment_cifar, attack_type, max_test_batches):

        # pixel-pattern backdoor augmentation
        pixel_cb = image_augmentation.pixel_pattern_if_needed(
            attack_type == "pixel_pattern"
        )

        if aux_size == 0:
            ds = (
                tf.data.Dataset.from_tensor_slices(
                    (self.x_aux_test, self.mal_aux_labels_test)
                )
                .batch(batch_size)
                .map(pixel_cb, num_parallel_calls=AUTOTUNE)
            )

            if max_test_batches is not None:
                ds = ds.shuffle(max_test_batches).take(max_test_batches)

            return ds.prefetch(AUTOTUNE)

        # Repeated aux dataset
        ds = (
            tf.data.Dataset.from_tensor_slices(
                (self.x_aux_test, self.mal_aux_labels_test)
            )
            .repeat(aux_size)
            .batch(batch_size)
            .map(pixel_cb, num_parallel_calls=AUTOTUNE)
        )

        if augment_cifar:
            ds = ds.map(image_augmentation.test_aux_augment, num_parallel_calls=AUTOTUNE)

        return ds.prefetch(AUTOTUNE)

    # ----------------------------------------------------------------------

    def get_full_dataset(self, size):
        """Returns a random subset of the global training dataset."""
        x = np.concatenate(self.x_train)
        y = np.concatenate(self.y_train)

        idx = np.random.choice(len(x), size, replace=False)
        return x[idx], y[idx]


# ----------------------------------------------------------------------
# IID dataset
# ----------------------------------------------------------------------

class IIDGlobalDataset(GlobalDataset):
    def __init__(self, x_train, y_train, num_clients, x_test, y_test):
        super().__init__(x_test, y_test)
        self.num_clients = num_clients
        x_train, y_train = self.shuffle(x_train, y_train)

        samples_per_client = len(x_train) // num_clients

        for cid in range(num_clients):
            lo = cid * samples_per_client
            hi = (cid + 1) * samples_per_client

            self.x_train.append(x_train[lo:hi])
            self.y_train.append(y_train[lo:hi])

    def shuffle(self, x, y):
        perm = np.random.permutation(len(x))
        return x[perm], y[perm]

    def get_dataset_for_client(self, client_id):
        return self.x_train[client_id], self.y_train[client_id]


# ----------------------------------------------------------------------
# Non-IID dataset
# ----------------------------------------------------------------------

class NonIIDGlobalDataset(GlobalDataset):
    def __init__(self, x_train, y_train, x_test, y_test, num_clients):
        super().__init__(x_test, y_test)
        self.x_train = x_train
        self.y_train = y_train

    def shuffle(self):
        raise Exception("Non-IID dataset cannot be shuffled.")

    def get_dataset_for_client(self, client_id):
        return self.x_train[client_id], self.y_train[client_id]


# ----------------------------------------------------------------------
# Dirichlet division
# ----------------------------------------------------------------------

class DirichletDistributionDivider:
    """Splits dataset by Dirichlet distribution."""

    def __init__(self, x_train, y_train,
                 train_aux, test_aux,
                 exclude_aux, num_clients):
        self.x_train = x_train
        self.y_train = y_train
        self.train_aux = train_aux
        self.test_aux = test_aux
        self.exclude_aux = exclude_aux
        self.num_clients = num_clients

    def build(self):
        alpha = 0.9
        per_class = defaultdict(list)

        for idx, label in enumerate(self.y_train):
            if self.exclude_aux and (idx in self.train_aux or idx in self.test_aux):
                continue
            per_class[label].append(idx)

        class_size = len(per_class[0])
        per_user = defaultdict(list)
        num_classes = len(per_class)

        for cls in range(num_classes):
            arr = per_class[cls]
            np.random.shuffle(arr)

            probs = class_size * np.random.dirichlet([alpha] * self.num_clients)

            for user_id in range(self.num_clients):
                count = int(round(probs[user_id]))
                take = min(len(arr), count)
                per_user[user_id].extend(arr[:take])
                arr = arr[take:]

        xs, ys = [], []
        for user_id in range(self.num_clients):
            inds = per_user[user_id]
            xs_user = self.x_train[inds]
            ys_user = self.y_train[inds]

            perm = np.random.permutation(len(xs_user))
            xs.append(xs_user[perm])
            ys.append(ys_user[perm])

        return xs, ys
