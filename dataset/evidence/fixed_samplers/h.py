import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IndexHiddenSampler:
    def __init__(self,
                 scm: TensorSCM,
                 max_length: int = 1,
                 ) -> None:
        self._scm = scm
        self._max_length = max_length

    def sample(self) -> int:
        return th.randint(0, self._max_length, th.Size()).item()

    def batched_sample(self, batch_size: int) -> th.Tensor:
        return th.randint(0, self._max_length, (batch_size, ), device=self._scm.device)
