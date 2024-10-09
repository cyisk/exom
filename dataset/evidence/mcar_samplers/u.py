import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IdenticalDistributedExogenousSampler:
    def __init__(self, scm: TensorSCM) -> None:
        self._scm = scm

    def sample(self, H: None) -> TensorDict:
        return self._scm.sample()

    def batched_sample(self, batch_size: int, h: th.Tensor) -> th.Tensor:
        U = self._scm.sample(batch_size)
        return batch(U, self._scm.exogenous_dimensions)
