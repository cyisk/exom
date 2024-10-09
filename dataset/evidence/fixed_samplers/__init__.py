import torch as th
from typing import *

from dataset.evidence.evidence_sampler import EvidenceSampler, TensorSCM, Evidence
from dataset.evidence.batched_evidence_sampler import BatchedEvidenceSampler, BatchedEvidence
from dataset.evidence.fixed_samplers.j import IndexJointNumberSampler
from dataset.evidence.fixed_samplers.h import IndexHiddenSampler
from dataset.evidence.mcar_samplers.u import IdenticalDistributedExogenousSampler
from dataset.evidence.fixed_samplers.itv import IndexIntervenedSampler
from dataset.evidence.fixed_samplers.obs import IndexObservedSampler
from dataset.evidence.fixed_samplers.s import IndexAntecedentSampler
from dataset.evidence.mcar_samplers.fobs import BernoulliFeatureObservedSampler
from common.scm.utils import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike',
                              bound=BatchedEvidence)


class FixedSampler(EvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 batches: str | List[th.Tensor] = None,
                 ) -> None:
        if isinstance(batches, str):
            batches = th.load(batches)
        assert batches is not None

        # Preprocess
        self._j_list = th.tensor(
            [th.sum(batch[0].int()).item() for batch in batches],
            device=self._scm.device,
        )
        self._max_len = th.max(self._j_list)
        self._t_list = th.tensor([
            batch[3] for batch in batches
        ])
        self._w_e_list = th.tensor([
            batch[2] for batch in batches
        ])
        self._w_t_list = th.tensor([
            batch[4] for batch in batches
        ])

        super().__init__(
            scm=scm,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_number_sampler=IndexJointNumberSampler(
                scm=scm,
                j_list=self._j_list
            ),
            joint_hidden_sampler=IndexHiddenSampler(
                scm=scm,
                max_length=len(batches)
            ),
            exogenous_sampler=IdenticalDistributedExogenousSampler(scm),
            antecedent_sampler=IndexAntecedentSampler(
                scm=scm,
                t_list=self._t_list,
            ),
            intervened_sampler=IndexIntervenedSampler(
                scm=scm,
                w_t_list=self._w_t_list,
            ),
            observed_sampler=IndexObservedSampler(
                scm=scm,
                w_e_list=self._w_e_list,
            ),
            feature_observed_sampler=BernoulliFeatureObservedSampler(
                scm=scm,
                prob=1,
            )
        )


class BatchedFixedSampler(BatchedEvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 batched_evidence_type: Type[BatchedEvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 batches: str | List[th.Tensor] = None,
                 ) -> None:
        if isinstance(batches, str):
            batches = th.load(batches)
        assert batches is not None

        # Preprocess
        self._j_list = th.tensor(
            [th.sum(batch[0].int()).item() for batch in batches],
            device=scm.device,
        )
        self._max_len = th.max(self._j_list)
        self._t_list = th.stack([
            batch[3] for batch in batches
        ], dim=0)
        self._w_e_list = th.stack([
            batch[2] for batch in batches
        ], dim=0)
        self._w_t_list = th.stack([
            batch[4] for batch in batches
        ], dim=0)

        # Squeeze mask
        W_E = {
            k: v.any(dim=[i for i in range(2, v.dim())])
            if v.dim() > 2 else v for k, v in
            unbatch(self._w_e_list, scm.endogenous_dimensions).items()
        }
        self._w_e_list = th.stack(
            [W_E[v] for v in sorted(scm.endogenous_variables)], dim=-1
        )
        W_T = {
            k: v.any(dim=[i for i in range(2, v.dim())])
            if v.dim() > 2 else v for k, v in
            unbatch(self._w_t_list, scm.endogenous_dimensions).items()
        }
        self._w_t_list = th.stack(
            [W_T[v] for v in sorted(scm.endogenous_variables)], dim=-1
        )

        super().__init__(
            scm=scm,
            batched_evidence_type=batched_evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_number_sampler=IndexJointNumberSampler(
                scm=scm,
                j_list=self._j_list
            ),
            joint_hidden_sampler=IndexHiddenSampler(
                scm=scm,
                max_length=len(batches)
            ),
            exogenous_sampler=IdenticalDistributedExogenousSampler(scm),
            antecedent_sampler=IndexAntecedentSampler(
                scm=scm,
                t_list=self._t_list,
            ),
            intervened_sampler=IndexIntervenedSampler(
                scm=scm,
                w_t_list=self._w_t_list,
            ),
            observed_sampler=IndexObservedSampler(
                scm=scm,
                w_e_list=self._w_e_list,
            ),
            feature_observed_sampler=BernoulliFeatureObservedSampler(
                scm=scm,
                prob=1.,
            )
        )
