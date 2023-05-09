import numpy as np
import tensorflow as tf


def load_Xy_from_numpy(np_dataset):
    # load files
    X = np.load(np_dataset[0])["x"]
    y = np.load(np_dataset[0])["y"]

    for np_file in np_dataset[1:]:
        X = np.vstack((X, np.load(np_file)["x"]))
        y = np.append(y, np.load(np_file)["y"])
    return X, y


def make_dataset(data, batch_size):
    print(data)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    print(dataset)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.shuffle(len(dataset), reshuffle_each_iteration=True)

    return dataset


def data_generator_np(training_files, subject_files, batch_size):
    X_train, y_train = load_Xy_from_numpy(training_files)
    X_test, y_test = load_Xy_from_numpy(subject_files)
    print(X_train.shape)
    print(y_train.shape)
    

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((y_train, y_test))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_dataset = make_dataset((X_train, y_train), batch_size)
    test_dataset = make_dataset((X_test, y_test), batch_size)

    return train_dataset, test_dataset, counts
