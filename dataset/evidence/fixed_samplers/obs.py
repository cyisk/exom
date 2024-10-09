import torch as th
from torch.distributions import Bernoulli
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IndexObservedSampler:
    def __init__(self, scm: TensorSCM, w_e_list: th.Tensor) -> None:
        self._scm = scm
        self._w_e_list = w_e_list

    def sample(self, h: None, j: None, U: TensorDict) -> TensorDict:
        contained = self._w_e_list[h, j, :]
        return set([self._endos[i] for i in range(len(self._endos)) if contained[i]])

    def batched_sample(self, h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
        return self._w_e_list.to(h.device)[h, j, :]
