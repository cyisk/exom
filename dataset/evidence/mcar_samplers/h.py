import torch as th
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class NoneJointHiddenSampler:
    def __init__(self,
                 scm: TensorSCM,
                 ) -> None:
        self._scm = scm

    def sample(self) -> int:
        return 0

    def batched_sample(self, batch_size: int) -> th.Tensor:
        return th.zeros((batch_size, ), device=self._scm.device)
