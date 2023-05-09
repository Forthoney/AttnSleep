import numpy as np
import tensorflow as tf

from base import BaseTrainer
from utils import MetricTracker, inf_loop

selected_d = {"outs": [], "trg": []}


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        data_loader,
        fold_id,
        valid_data_loader=None,
        class_weights=None,
        batch_size=128,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch


        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.train_metrics.reset()  # Reset the training metrics
        overall_outs = []
        overall_trgs = []

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = (
                data.numpy(),
                target.numpy(),
            )  # Convert tensor data to numpy arrays
            data = tf.convert_to_tensor(
                data, dtype=tf.float32
            )  # Convert numpy arrays to tensorflow tensors
            target = tf.convert_to_tensor(target, dtype=tf.int32)
            target = tf.one_hot(target, depth=5)

            with tf.GradientTape() as tape:
                output = self.model(data)
                loss = self.criterion(output, target)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            self.train_metrics.update("loss", loss)
            target = tf.cast(target, dtype=tf.int64)
            output = tf.cast(output, dtype=tf.int64)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f} ".format(
                        epoch,
                        self._progress(batch_idx),
                        loss.numpy(),
                    )
                )

            if batch_idx == self.len_epoch:
                break
        log = {m.name: m.result().numpy() for m in self.train_metrics.metrics}

        if self.do_validation:
            val_log, outs, trgs = self.valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # The following part is used to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                self.lr_scheduler.learning_rate = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch


        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.trainable = False  # Set model to evaluation mode
        self.valid_metrics.reset_states()  # Reset metric states
        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data, training=False)
            loss = self.criterion(output, target, self.class_weights, self.device)

            self.valid_metrics.update("loss", loss.numpy())
            for met in self.metric_ftns:
                self.valid_metrics.update(
                    met.__name__, met(output.numpy(), target.numpy())
                )

            preds_ = tf.argmax(output, axis=1)

            outs = np.append(outs, preds_.numpy())
            trgs = np.append(trgs, target.numpy())

        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
