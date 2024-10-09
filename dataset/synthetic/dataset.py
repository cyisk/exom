import torch as th
from torch.utils.data import Dataset
from typing import Dict, Optional

from common.scm import TensorSCM, batch


class ObservationalDataset(Dataset):
    def __init__(self,
                 scm: TensorSCM,
                 size: int = 16384,
                 ) -> None:
        super().__init__()
        self._scm = scm
        self._size = size

        self.initialize()

    def initialize(self) -> None:
        self._u = self._scm.batched_sample(self._size)
        self._v = self._scm.batched_call(self._u)

        self._v_mean = self._v.detach().mean(dim=0)
        self._v_std = self._v.detach().std(dim=0)

    def __getitem__(self, index) -> th.Tensor:
        return self._v[index]

    def __len__(self) -> int:
        return self._size

    @property
    def mean(self) -> th.Tensor:
        return self._v_mean

    @property
    def std(self) -> th.Tensor:
        return self._v_std
