from dataset.evidence.evidence import Evidence
from dataset.evidence.evidence_joint import EvidenceJoint
from dataset.evidence.evidence_custom import EvidenceContextConcat, EvidenceContextMasked
from dataset.evidence.evidence_sampler import EvidenceSampler

from dataset.evidence.batched_evidence import BatchedEvidence
from dataset.evidence.batched_evidence_joint import BatchedEvidenceJoint
from dataset.evidence.batched_evidence_custom import BatchedEvidenceContextConcat, BatchedEvidenceContextMasked
from dataset.evidence.batched_evidence_sampler import BatchedEvidenceSampler


from typing import TypeVar
EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike', bound=BatchedEvidence)


def to_batched_evidence(evidence_type: EvidenceLike):
    return {
        Evidence: BatchedEvidence,
        EvidenceJoint: BatchedEvidenceJoint,
        EvidenceContextConcat: BatchedEvidenceContextConcat,
        EvidenceContextMasked: BatchedEvidenceContextMasked,
    }[evidence_type]


def to_unbatched_evidence(evidence_type: BatchedEvidenceLike):
    return {
        BatchedEvidence: Evidence,
        BatchedEvidenceJoint: EvidenceJoint,
        BatchedEvidenceContextConcat: EvidenceContextConcat,
        BatchedEvidenceContextMasked: EvidenceContextMasked,
    }[evidence_type]


def to_batched_sampler(sampler: EvidenceSampler):
    # This will assume that each sampler also allows batched sampling
    return BatchedEvidenceSampler(
        sampler._scm,
        to_batched_evidence(sampler._evidence_type),
        sampler._evidence_kwargs,
        sampler._h_sampler,
        sampler._j_sampler,
        sampler._u_sampler,
        sampler._s_sampler,
        sampler._itv_sampler,
        sampler._obs_sampler,
        sampler._fobs_sampler,
    )


def to_unbatched_sampler(sampler: BatchedEvidenceSampler):
    # This will assume that each sampler also allows unbatched sampling
    return EvidenceSampler(
        sampler._scm,
        to_unbatched_evidence(sampler._batched_evidence_type),
        sampler._evidence_kwargs,
        sampler._h_sampler,
        sampler._j_sampler,
        sampler._u_sampler,
        sampler._s_sampler,
        sampler._itv_sampler,
        sampler._obs_sampler,
        sampler._fobs_sampler,
    )
