import json
from argparse import ArgumentParser

import tensorflow as tf

from load_data import data_generator_np
from model.model import AttnSleep
from util import calc_class_weight, load_folds_data, load_folds_data_shhs

# fix random seeds for reproducibility
SEED = 123
tf.random.set_seed(SEED)


def load_data(num_folds: int, data_dir: str):
    if "shhs" in data_dir:
        folds_data = load_folds_data_shhs(data_dir, num_folds)
    else:
        folds_data = load_folds_data(data_dir, num_folds)

    return folds_data


def weights_init_normal(layer: tf.keras.layers.Layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.kernel_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02
        )
        if layer.bias is not None:
            layer.bias_initializer = tf.keras.initializers.Zeros()
    elif isinstance(layer, tf.keras.layers.Conv1D):
        layer.kernel_initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.02
        )
        if layer.bias is not None:
            layer.bias_initializer = tf.keras.initializers.Zeros()
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.gamma_initializer = tf.keras.initializers.RandomNormal(
            mean=1.0, stddev=0.02
        )
        layer.beta_initializer = tf.keras.initializers.Zeros()


def prepare_datasets(num_folds, data_dir, fold_id, batch_size):
    dataset = load_data(num_folds, data_dir)
    train_data, val_data, counts = data_generator_np(
        dataset[fold_id][0], dataset[fold_id][1], batch_size
    )
    class_weight = calc_class_weight(counts)
    return train_data, val_data, class_weight


if __name__ == "__main__":
    parser = ArgumentParser("AttnSleep Training")
    parser.add_argument("-c", "--config", default="config.json", type=str)
    parser.add_argument("--checkpoint", default="checkpoints", type=str)
    parser.add_argument("-f", "--fold_id", type=int)
    parser.add_argument("-da", "--data_dir", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    train_data, val_data, class_weight = prepare_datasets(
        config["num_folds"], args.data_dir, args.fold_id, config["batch_size"]
    )

    model = AttnSleep()
    for layer in model.layers:
        weights_init_normal(layer)
    loss = tf.keras.losses.get(config["loss"])
    optimizer = tf.keras.optimizers.get(config["optimizer"])
    metrics = [tf.keras.metrics.get(metric) for metric in config["metrics"]]

    model.compile(optimizer, loss, metrics, run_eagerly=False, jit_compile=True)

    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.checkpoint,  # path to save the checkpoint file
        save_weights_only=True,  # save only the weights instead of the whole model
        monitor="val_categorical_accuracy",  # metric to monitor for saving the checkpoint
        mode="max",  # mode of the monitored metric (max or min)
        save_best_only=True,  # save only the best model according to the monitored metric
    )

    model.fit(
        train_data,
        epochs=config["num_epochs"],
        validation_data=val_data,
        callbacks=[checkpointing],
        class_weight=class_weight,
    )
