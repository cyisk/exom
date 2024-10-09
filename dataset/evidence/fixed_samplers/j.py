import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class IndexJointNumberSampler:
    def __init__(self,
                 scm: TensorSCM,
                 j_list: th.Tensor,
                 ) -> None:
        self._scm = scm
        self._j_list = j_list

    def sample(self, H: None) -> int:
        return self._j_list[H]

    def batched_sample(self, batch_size: int, h: th.Tensor) -> th.Tensor:
        assert h.size(0) == batch_size
        return self._j_list[h]
