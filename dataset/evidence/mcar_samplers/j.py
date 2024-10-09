import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class UniformJointNumberSampler:
    def __init__(self,
                 scm: TensorSCM,
                 low: int = 1,
                 high: int = 1,
                 ) -> None:
        self._scm = scm
        self._low = low
        self._high = high

    def sample(self, H: None) -> int:
        return th.randint(self._low, self._high + 1, th.Size(), device=self._scm.device)

    def batched_sample(self, batch_size: int, h: th.Tensor) -> th.Tensor:
        return th.randint(self._low, self._high + 1, (batch_size, ), device=self._scm.device)
