import torch as th
from typing import *

from common.scm import *
from dataset.evidence.evidence import *


TensorDict = Dict[str, th.Tensor]


class EvidenceJoint:
    def __init__(self, evidences: List[Evidence]) -> None:
        # Assume all evidences is from the same SCM and has solveability w.r.t exogenous variables
        self._evidences = evidences

    def __getitem__(self, index) -> Evidence:
        return self._evidences[index]

    def __len__(self) -> int:
        return len(self._evidences)

    def stack_object(self,
                     objects: Iterable[Any],
                     max_len: int = -1,
                     ) -> Tuple[th.Tensor]:
        if len(self) == 0:
            return None
        if max_len == -1:
            max_len = len(self)
        dum_len = max_len - len(self)

        return [
            objects[i]
            for i in range(len(self))
            if i < max_len
        ] + [None] * dum_len

    def stack_tensor(self,
                     tensors: Iterable[th.Tensor | None],
                     max_len: int = -1,
                     ) -> Tuple[th.Tensor]:
        dummy_tensor = th.zeros_like(tensors[0])
        padding_len = max(0, max_len - len(tensors))
        return th.stack(tensors + [dummy_tensor] * padding_len, dim=0)

    def stack(self,
              attr_name: str,
              max_len: int = -1,
              ) -> Any:
        attrs = self.get_attrs(attr_name)
        if all(isinstance(attr, th.Tensor) for attr in attrs):
            return self.stack_tensor(attrs, max_len)
        else:
            return self.stack_object(attrs, max_len)

    def get_attrs(self, attr_name: str) -> List[Any]:
        if len(self) == 0:
            return None
        return [getattr(evidence, attr_name) for evidence in self]
