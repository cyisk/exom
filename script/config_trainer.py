from dataclasses import dataclass
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from typing import *

from script.config import Config


@dataclass
class DefaultTrainerConfig(Config):
    # Basics
    name: str = 'experiment'
    accelerator: str = 'auto'
    # Epochs
    max_epochs: int = -1
    check_val_every_n_epoch: int = 1
    progress_bar_enable: bool = False
    # Checkpoint (end of training)
    checkpoint_enable: bool = False
    checkpoint_name: str = None
    # Logger
    logger_enable: bool = False
    # Early stopping
    early_stop_monitor: str = None
    early_stop_mode: int = None
    early_stop_patience: int = None
    # Graident Clipping
    gradient_clipping: float = None

    def __post_init__(self):
        self.progress_bar_enable = bool(self.progress_bar_enable)
        self.checkpoint_enable = bool(self.checkpoint_enable)
        self.logger_enable = bool(self.logger_enable)
        if self.checkpoint_name is None:
            self.checkpoint_name = 'checkpoint'

    def get_trainer(self, callbacks: Iterable[Callback] = None) -> Trainer:
        if callbacks is None:
            callbacks = []

        # Logger
        if self.logger_enable:
            logger = TensorBoardLogger(
                save_dir='output/',
                name=self.name + '/logs',
            ),
        else:
            logger = False

        # Checkpoint
        if self.checkpoint_enable:
            checkpoint_callback = ModelCheckpoint(
                dirpath='output/',
                filename=self.name + '/checkpoints/' + self.checkpoint_name,
            )
            callbacks.append(checkpoint_callback)

        # Early stopping
        if self.early_stop_monitor is not None:
            early_stop_callback = EarlyStopping(
                monitor=self.early_stop_monitor,
                mode=self.early_stop_mode,
                patience=self.early_stop_patience,
            )
            callbacks.append(early_stop_callback)

        return Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            enable_progress_bar=self.progress_bar_enable,
            logger=logger,
            enable_checkpointing=self.checkpoint_enable,
            callbacks=callbacks,
            gradient_clip_val=self.gradient_clipping,
        )
