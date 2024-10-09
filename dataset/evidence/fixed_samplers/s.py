import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IndexAntecedentSampler:
    def __init__(self, scm: TensorSCM, t_list: th.Tensor) -> None:
        self._scm = scm
        self._t_list = t_list

    def sample(self, h: None, j: None, U: TensorDict) -> TensorDict:
        return unbatch(self._t_list[h, j, :], self._scm.endogenous_dimensions)

    def batched_sample(self, h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self._t_list.to(h.device)[h, j, :]
