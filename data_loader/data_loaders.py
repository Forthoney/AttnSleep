import numpy as np
import tensorflow as tf


def load_Xy_from_numpy(np_dataset: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # load files
    X = tuple([np.load(datum)["x"] for datum in np_dataset])
    y = tuple([np.load(datum)["y"] for datum in np_dataset])

    return np.vstack(X), np.concatenate(y)


def make_dataset(
    data: tuple[np.ndarray, np.ndarray], batch_size: int
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True)

    return dataset


def data_generator_np(
    training_files: list[str], subject_files: list[str], batch_size: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[int]]:
    X_train, y_train = load_Xy_from_numpy(training_files)
    X_test, y_test = load_Xy_from_numpy(subject_files)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((y_train, y_test))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    y_train = tf.one_hot(y_train, depth=5)
    y_test = tf.one_hot(y_test, depth=5)

    train_dataset = make_dataset((X_train, y_train), batch_size)
    test_dataset = make_dataset((X_test, y_test), batch_size)

    return train_dataset, test_dataset, counts
