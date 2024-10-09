import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IdenticalDistributedAntecedentSampler:
    def __init__(self, scm: TensorSCM) -> None:
        self._scm = scm

    def sample(self, h: None, j: None, U: TensorDict) -> TensorDict:
        U_ = self._scm.sample()
        return self._scm(U_)

    def batched_sample(self, h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
        U_ = self._scm.sample(u.size(0))
        S_ = self._scm(U_)
        return batch(S_, self._scm.endogenous_dimensions)
