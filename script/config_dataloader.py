from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from typing import *

from script.config import Config


@dataclass
class DefaultDataloaderConfig(Config):
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False

    def __post_init__(self):
        self.shuffle = bool(self.shuffle)
        self.pin_memory = bool(self.pin_memory)
        self.drop_last = bool(self.drop_last)

    def get_datasetloader(self,
                          dataset: Dataset,
                          collate_fn: Callable = None,
                          worker_init_fn: Callable = None,
                          ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )
