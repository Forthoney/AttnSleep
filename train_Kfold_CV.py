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


def weights_init_normal(layer):
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


def main(config, fold_id):
    batch_size = config["data_loader"]["args"]["batch_size"]
    print("batch_size", batch_size)

    logger = config.get_logger("train")

    # build model architecture, initialize weights, then print to console
    model: tf.keras.Model = config.init_obj("arch", module_arch)
    for layer in model.layers:
        weights_init_normal(layer)
    logger.info(model)

    print("Building Model...")
    model.build((batch_size, 3000, 1))
    model.summary(expand_nested=True, show_trainable=True)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    print("Loading Data...")
    data_loader, valid_data_loader, data_count = data_generator_np(
        folds_data[fold_id][0], folds_data[fold_id][1], batch_size
    )
    weights_for_each_class = calc_class_weight(data_count)

    optimizer = config.init_obj("optimizer", tf.optimizers)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        fold_id=fold_id,
        valid_data_loader=valid_data_loader,
        class_weights=weights_for_each_class,
    )

    print("Starting training")

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Tensorflow Template")
    args.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default="0",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument("-f", "--fold_id", type=str, help="fold_id")
    args.add_argument(
        "-da", "--np_data_dir", type=str, help="Directory containing numpy files"
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(
            args2.np_data_dir, config["data_loader"]["args"]["num_folds"]
        )
    else:
        folds_data = load_folds_data(
            args2.np_data_dir, config["data_loader"]["args"]["num_folds"]
        )

    main(config, fold_id)
