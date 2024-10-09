import torch as th
from typing import *

from common.scm import *
from dataset.utils import *
from dataset.evidence.evidence_joint import *
from dataset.evidence.batched_evidence import *


TensorDict = Dict[str, th.Tensor]


class BatchedEvidenceJoint:
    def __init__(self,
                 batched_evidence: BatchedEvidence,
                 joint_length: th.Tensor) -> None:
        assert th.sum(joint_length) == len(batched_evidence)
        self._batched_evidence = batched_evidence
        self._joint_length = joint_length
        self._cumsum_joint_length = th.cat(
            (th.tensor([0]), th.cumsum(joint_length, dim=0)), dim=0
        )

        # Unflatten joint mask
        batch_size = len(joint_length)
        total_batch_size = len(batched_evidence)
        max_joint_size = th.max(joint_length)
        joint_idcs = th.arange(max_joint_size)[None, :].expand(batch_size, -1)
        w_j = joint_idcs < joint_length[:, None].expand(-1, max_joint_size)

        # Masked scatter
        j_e = feature_masked_scatter(batched_evidence.e, w_j)
        j_t = feature_masked_scatter(batched_evidence.t, w_j)
        j_w_e = feature_masked_scatter(batched_evidence.w_e, w_j)
        j_w_t = feature_masked_scatter(batched_evidence.w_t, w_j)

        # Set buffer
        self._batch_size = batch_size
        self._total_batch_size = total_batch_size
        self._max_joint_size = max_joint_size
        self._w_j = w_j
        self._j_e = j_e
        self._j_t = j_t
        self._j_w_e = j_w_e
        self._j_w_t = j_w_t

    def __getitem__(self, index) -> Evidence:
        start_i = self._cumsum_joint_length[index]
        return EvidenceJoint([
            self._batched_evidence[start_i + i]
            for i in range(self._joint_length[index])
        ])

    def __len__(self) -> int:
        return self._batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def total_size(self) -> int:
        return self._total_batch_size

    @property
    def joint_size(self) -> int:
        return self._max_joint_size

    @property
    def w_j(self) -> th.Tensor:
        return self._w_j

    @property
    def e(self) -> th.Tensor:
        return self._j_e

    @property
    def t(self) -> th.Tensor:
        return self._j_t

    @property
    def w_e(self) -> th.Tensor:
        return self._j_w_e

    @property
    def w_t(self) -> th.Tensor:
        return self._j_w_t

    def get_context(self, index: int) -> th.Tensor | None:
        start_i = self._cumsum_joint_length[index]
        contexts = [
            self._batched_evidence.get_context(start_i + i)
            for i in range(self._joint_length[index])
        ]
        if any(context is None for context in contexts):
            return None
        zero = th.zeros_like(contexts[0])
        contexts = contexts + [zero] * (self.joint_size - len(contexts))
        return th.stack(contexts, dim=0)

    def get_adjacency(self, index: int) -> th.Tensor | None:
        start_i = self._cumsum_joint_length[index]
        adjacencies = [
            self._batched_evidence.get_adjacency(start_i + i)
            for i in range(self._joint_length[index])
        ]
        if any(adjacency is None for adjacency in adjacencies):
            return None
        zero = th.zeros_like(adjacencies[0])  # False for padding
        adjacencies = adjacencies + [zero] * \
            (self.joint_size - len(adjacencies))
        return th.stack(adjacencies, dim=0)
