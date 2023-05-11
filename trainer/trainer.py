import numpy as np
import tensorflow as tf

from parse_config import ConfigParser
from utils import MetricTracker, inf_loop

selected_d = {"outs": [], "trg": []}


class Trainer:
    """
    Trainer class
    """

    def __init__(
        self,
        model: tf.keras.Model,
        criterion: tf.keras.losses.Loss,
        metric_ftns: list[tf.keras.metrics.Metric],
        optimizer: tf.keras.optimizers.Optimizer,
        config: ConfigParser,
        data_loader: tf.data.Dataset,
        fold_id: int,
        valid_data_loader: tf.data.Dataset | None = None,
        class_weights: list[int] | None = None,
        batch_size: int = 128,
    ):
        self.config = config
        self.device, device_ids = self._prepare_device(config["n_gpu"])

        self.model = model
        self.loss = criterion
        self.metric_functions = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])
        self.n_epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.fold_id = fold_id

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = bool(self.valid_data_loader)
        self.lr_scheduler = optimizer
        self.log_step = int(batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_functions]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_functions]
        )

        self.selected = 0
        self.class_weights = class_weights

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = np.inf if self.mnt_mode == "min" else -np.inf
            self.early_stop = cfg_trainer.get("early_stop", np.inf)
        self.checkpoint_dir = config.save_dir

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        all_outs = []
        all_trgs = []

        # Not very confident in checkpoint functionality ngl
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        for epoch in range(1, self.n_epochs + 1):
            result = self._train_epoch(epoch, self.n_epochs)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            if self.do_validation:
                selected_d = {"outs": [], "trg": []}
                val_log, outs, trgs = self._valid_epoch(epoch)
                log.update(**{"val_" + k: v for k, v in val_log.items()})
                if val_log["accuracy"] > self.selected:
                    self.selected = val_log["accuracy"]
                    selected_d["outs"] = outs
                    selected_d["trg"] = trgs
                if epoch == self.n_epochs:
                    all_outs.extend(selected_d["outs"])
                    all_trgs.extend(selected_d["trg"])
            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False  # True
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self.loss(checkpoint, epoch, save_best=best)

        outs_name = "outs_" + str(self.fold_id)
        trgs_name = "trgs_" + str(self.fold_id)
        np.save(self.config._save_dir / outs_name, all_outs)
        np.save(self.config._save_dir / trgs_name, all_trgs)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            self._calc_metrics()

    def _train_epoch(self, epoch: int, total_epochs: int):
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
            with tf.GradientTape() as tape:
                output = self.model(data)
                loss = self.loss(output, target)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            self.train_metrics.update("loss", loss)
            target = tf.cast(target, dtype=tf.int64)
            output = tf.cast(output, dtype=tf.int64)

            for met in self.metric_functions:
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
        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
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
            self.lr_scheduler.learning_rate = 1e-4

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch


        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.trainable = False  # Set model to evaluation mode
        self.valid_metrics.reset()  # Reset metric states
        outs = np.array([])
        trgs = np.array([])
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            output = self.model(data, training=False)
            loss = self.loss(output, target)

            self.valid_metrics.update("loss", loss.numpy())
            for met in self.metric_functions:
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

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """

        n_gpus = len(tf.config.list_physical_devices("GPU"))
        if n_gpu_use > 0 and n_gpus == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpus:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpus)
            )
            n_gpu_use = n_gpus

        device = tf.device("/gpu:0" if n_gpu_use > 0 else "/cpu:0")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def loss(self, checkpoint: tf.train.Checkpoint, epoch, save_best=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))

        checkpoint.save(filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            checkpoint.save(best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, checkpoint: tf.train.Checkpoint, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint.restore(resume_path)

    def _calc_metrics(self):
        import os

        import pandas as pd
        from sklearn.metrics import (accuracy_score, classification_report,
                                     cohen_kappa_score, confusion_matrix)

        n_folds = self.config["data_loader"]["args"]["num_folds"]
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                    outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                    trgs_list.append(os.path.join(root, file))

        if len(outs_list) == self.config["data_loader"]["args"]["num_folds"]:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        np.save(cm_Save_path, cm)
