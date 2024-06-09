import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger

from plcls.data import TorchVisionDataModule
from plcls.ema import EMACallback
from plcls.train import ClassifierModel


def prepare_environment(conf):
    L.seed_everything(conf.common.seed, workers=True)
    torch.set_float32_matmul_precision(conf.common.mm_precision)


def get_callbacks(conf):
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(**conf.model_checkpoint),
        LearningRateMonitor(logging_interval="step"),
    ]

    if conf.boosting.swa.enable:
        callbacks.append(
            StochasticWeightAveraging(
                conf.boosting.swa.swa_lrs,
                annealing_epochs=conf.boosting.swa.annealing_epochs,
                swa_epoch_start=conf.boosting.swa.swa_epoch_start,
                annealing_strategy=conf.boosting.swa.annealing_strategy,
            )
        )

    if conf.boosting.ema.enable:
        callbacks.append(EMACallback(decay=conf.boosting.ema.decay))

    return callbacks


def get_logger(conf):
    logger = TensorBoardLogger(**conf.logger)
    return logger


def train(conf):
    prepare_environment(conf)

    model = ClassifierModel(conf=conf)

    if conf.common.lightning_checkpoint:
        model = model.load_from_checkpoint(conf.common.lightning_checkpoint)

    data = TorchVisionDataModule(conf)
    trainer = L.Trainer(
        callbacks=get_callbacks(conf),
        logger=get_logger(conf),
        **conf.trainer.trainer_params,
    )
    trainer.fit(model, data)
