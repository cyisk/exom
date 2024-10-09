import abc
import torch as th
from typing import *

from common.scm import *
from dataset.evidence.evidence import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class BatchedEvidence(abc.ABC):
    def __init__(self,
                 scm: TensorSCM,
                 e_batched: th.Tensor,
                 t_batched: th.Tensor,
                 w_e_batched: th.BoolTensor,
                 w_t_batched: th.BoolTensor,
                 ) -> None:
        self._scm = scm

        # Shape check
        batch_shape = e_batched.shape[:-1]
        assert len(batch_shape) == 1, \
            "Expect batch size is the first dim"
        self._batch_size = batch_shape[0]

        # Batched information
        e_batched[w_t_batched] == t_batched[w_t_batched]
        self._e_batched = e_batched
        self._t_batched = t_batched
        self._w_e_batched = w_e_batched | w_t_batched
        self._w_t_batched = w_t_batched

        # Assure masked
        self._e_batched[~self._w_e_batched] = 0
        self._t_batched[~self._w_t_batched] = 0

    def __len__(self) -> int:
        return self._batch_size

    def __getitem__(self, index: int) -> Evidence:
        return Evidence(
            e=self._e_batched[index],
            t=self._t_batched[index],
            w_e=self._w_e_batched[index],
            w_t=self._w_t_batched[index],
        )

    @property
    def scm(self) -> TensorSCM:
        return self._scm

    @property
    def e(self) -> th.Tensor:
        return self._e_batched

    @property
    def t(self) -> th.Tensor:
        return self._t_batched

    @property
    def w_e(self) -> th.BoolTensor:
        return self._w_e_batched

    @property
    def w_t(self) -> th.BoolTensor:
        return self._w_t_batched

    @abc.abstractmethod
    def get_context(self, index: int) -> th.Tensor | None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_adjacency(self, index: int) -> th.Tensor | None:
        raise NotImplementedError

    @staticmethod
    def unbatched_type() -> Type[EvidenceLike]:
        return Evidence

    @staticmethod
    def context_features(scm: TensorSCM, *args, **kwargs) -> int:
        raise NotImplementedError

    @staticmethod
    def standardize(context: th.Tensor,
                    w_e: th.Tensor,
                    w_t: th.Tensor,
                    prior_mean: th.Tensor,
                    prior_std: th.Tensor,
                    *args,
                    **kwargs) -> th.Tensor:
        raise NotImplementedError
