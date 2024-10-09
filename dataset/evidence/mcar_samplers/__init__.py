from typing import *

from dataset.evidence.evidence_sampler import EvidenceSampler, TensorSCM, Evidence
from dataset.evidence.batched_evidence_sampler import BatchedEvidenceSampler, BatchedEvidence
from dataset.evidence.mcar_samplers.j import UniformJointNumberSampler
from dataset.evidence.mcar_samplers.h import NoneJointHiddenSampler
from dataset.evidence.mcar_samplers.u import IdenticalDistributedExogenousSampler
from dataset.evidence.mcar_samplers.itv import BernoulliIntervenedSampler
from dataset.evidence.mcar_samplers.obs import BernoulliObservedSampler
from dataset.evidence.mcar_samplers.s import IdenticalDistributedAntecedentSampler
from dataset.evidence.mcar_samplers.fobs import BernoulliFeatureObservedSampler

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike',
                              bound=BatchedEvidence)


class MCARBernoulliSampler(EvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 joint_number_low: int = 1,
                 joint_number_high: int = 1,
                 prob_intervened: float = 0.2,
                 prob_observed: float = 0.8,
                 prob_feature_observed: float = 1.0,
                 ) -> None:
        super().__init__(
            scm=scm,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_number_sampler=UniformJointNumberSampler(
                scm=scm,
                low=joint_number_low,
                high=joint_number_high,
            ),
            joint_hidden_sampler=NoneJointHiddenSampler(scm),
            exogenous_sampler=IdenticalDistributedExogenousSampler(scm),
            antecedent_sampler=IdenticalDistributedAntecedentSampler(scm),
            intervened_sampler=BernoulliIntervenedSampler(
                scm=scm,
                prob=prob_intervened,
            ),
            observed_sampler=BernoulliObservedSampler(
                scm=scm,
                prob=prob_observed,
            ),
            feature_observed_sampler=BernoulliFeatureObservedSampler(
                scm=scm,
                prob=prob_feature_observed,
            ),
        )


class BatchedMCARBernoulliSampler(BatchedEvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 batched_evidence_type: Type[BatchedEvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 joint_number_low: int = 1,
                 joint_number_high: int = 1,
                 prob_intervened: float = 0.2,
                 prob_observed: float = 0.8,
                 prob_feature_observed: float = 1.0,
                 ) -> None:
        super().__init__(
            scm=scm,
            batched_evidence_type=batched_evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_number_sampler=UniformJointNumberSampler(
                scm=scm,
                low=joint_number_low,
                high=joint_number_high,
            ),
            joint_hidden_sampler=NoneJointHiddenSampler(scm),
            exogenous_sampler=IdenticalDistributedExogenousSampler(scm),
            antecedent_sampler=IdenticalDistributedAntecedentSampler(scm),
            intervened_sampler=BernoulliIntervenedSampler(
                scm=scm,
                prob=prob_intervened,
            ),
            observed_sampler=BernoulliObservedSampler(
                scm=scm,
                prob=prob_observed,
            ),
            feature_observed_sampler=BernoulliFeatureObservedSampler(
                scm=scm,
                prob=prob_feature_observed,
            ),
        )
