import argparse
import collections

import numpy as np
import tensorflow as tf

import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader.data_loaders import *
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *

# fix random seeds for reproducibility
SEED = 123
tf.random.set_seed(SEED)
np.random.seed(SEED)


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


def main(config: ConfigParser, fold_id: int, data):
    batch_size = config["data_loader"]["args"]["batch_size"]
    logger = config.get_logger("train")

    # build model architecture, initialize weights, then print to console
    model: tf.keras.Model = config.init_obj("arch", module_arch)
    for layer in model.layers:
        weights_init_normal(layer)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    print("Loading Data...")
    train_data, val_data, data_count = data_generator_np(
        data[fold_id][0], data[fold_id][1], batch_size
    )

    optimizer = config.init_obj("optimizer", tf.optimizers)

    print("Starting training...")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=False,
        jit_compile=True,
    )
    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_checkpoint",  # path to save the checkpoint file
        save_weights_only=True,  # save only the weights instead of the whole model
        monitor="val_accuracy",  # metric to monitor for saving the checkpoint
        mode="max",  # mode of the monitored metric (max or min)
        save_best_only=True,  # save only the best model according to the monitored metric
    )
    model.fit(
        train_data, epochs=10, validation_data=val_data, callbacks=[checkpointing]
    )


def load_data(config, data_dir):
    if "shhs" in data_dir:
        folds_data = load_folds_data_shhs(
            data_dir, config["data_loader"]["args"]["num_folds"]
        )
    else:
        folds_data = load_folds_data(
            data_dir, config["data_loader"]["args"]["num_folds"]
        )

    return folds_data


def parse_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tensorflow Template")
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument("-f", "--fold_id", type=str, help="fold_id")
    parser.add_argument(
        "-da", "--np_data_dir", type=str, help="Directory containing numpy files"
    )

    return parser


if __name__ == "__main__":
    parser = parse_cli()
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = []

    args = parser.parse_args()
    fold_id = int(args.fold_id)
    config = ConfigParser.from_args(parser, fold_id, options)
    data = load_data(config, args.np_data_dir)

    main(config, fold_id, data)
