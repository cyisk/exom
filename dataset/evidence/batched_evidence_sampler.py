import torch as th
from typing import *

from common.scm import *
from dataset.utils import *
from dataset.evidence.evidence import *
from dataset.evidence.evidence_joint import *
from dataset.evidence.batched_evidence import *
from dataset.evidence.batched_evidence_joint import *

TensorDict = Dict[str, th.Tensor]


class _BatchedTensorSampler:
    def __init__(self, *args, **kwargs):
        pass

    def batched_sample(self, *args, **kwargs) -> th.Tensor:
        pass


BatchedTensorSampler = TypeVar('BatchedTensorSampler',
                               bound=_BatchedTensorSampler)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike',
                              bound=BatchedEvidence)


class BatchedEvidenceSampler:
    """
    Sample information needed in EvidenceJoint in one go.
    Returns BatchedEvidence and joint sizes.
    """

    def __init__(self,
                 scm: TensorSCM,
                 batched_evidence_type: Type[BatchedEvidenceLike],
                 evidence_kwargs: Dict[str, Any],
                 joint_hidden_sampler: BatchedTensorSampler,
                 joint_number_sampler: BatchedTensorSampler,
                 exogenous_sampler: BatchedTensorSampler,
                 antecedent_sampler: BatchedTensorSampler,
                 intervened_sampler: BatchedTensorSampler,
                 observed_sampler: BatchedTensorSampler,
                 feature_observed_sampler: BatchedTensorSampler,
                 ) -> None:
        self._scm = scm
        self._batched_evidence_type = batched_evidence_type
        self._evidence_kwargs = evidence_kwargs
        self._h_sampler = joint_hidden_sampler
        self._j_sampler = joint_number_sampler
        self._u_sampler = exogenous_sampler
        self._s_sampler = antecedent_sampler
        self._itv_sampler = intervened_sampler
        self._obs_sampler = observed_sampler
        self._fobs_sampler = feature_observed_sampler

    def batched_sample(self,
                       batch_size: int,
                       return_exogenous: bool = False,
                       ) -> Tuple[BatchedEvidenceJoint] | Tuple[th.Tensor, BatchedEvidenceJoint]:
        # Sample shared hidden state for each joint inference
        h = self._h_sampler.batched_sample(batch_size)

        # Sample batched joint inference sizes
        j = self._j_sampler.batched_sample(batch_size, h)

        # Sample batched exogenous
        u = self._u_sampler.batched_sample(batch_size, h)

        # Unflatten joint mask
        total_batch_size = th.sum(j)
        max_joint_size = th.max(j)
        joint_idcs = th.arange(max_joint_size, device=u.device)
        joint_idcs = joint_idcs[None, :].expand(batch_size, -1)
        w_j = joint_idcs < j[:, None].expand(-1, max_joint_size)

        # Flatten states for joint inference
        flatten_h = h[:, None, ...].expand(
            -1, max_joint_size, *([-1]*(h.dim()-1))
        )
        flatten_h = feature_masked_select(flatten_h, w_j)
        assert flatten_h.size(0) == total_batch_size

        # Flatten joint index for joint inference
        flatten_j = feature_masked_select(joint_idcs, w_j)
        assert flatten_h.size(0) == total_batch_size

        # Flatten exogenous for joint inference
        flatten_u = u[:, None, :].expand(-1, max_joint_size, -1)
        flatten_u = feature_masked_select(flatten_u, w_j)
        assert flatten_u.size(0) == total_batch_size

        # Sample non-missing indicators: [batch_size, |V|]
        intervened = self._itv_sampler.batched_sample(
            flatten_h, flatten_j, flatten_u
        )
        observed = self._obs_sampler.batched_sample(
            flatten_h, flatten_j, flatten_u
        )
        observed |= intervened  # intervened is also observed

        # Sample antecedent features: [batch_size, feature_size]
        s = self._s_sampler.batched_sample(flatten_h, flatten_j, flatten_u)

        # Sample observed features indicators: [batch_size, feature_size]
        feature_observed = self._fobs_sampler.batched_sample(
            observed, flatten_h, flatten_j, flatten_u
        )

        # t, w_t and w_e
        endo_features = th.tensor(
            list(self._scm.endogenous_features.values()),
            device=u.device,
        )
        t = s
        w_t = intervened.repeat_interleave(endo_features, dim=1)
        t[~w_t] = 0
        w_e = observed.repeat_interleave(endo_features, dim=1)
        w_e = w_e & feature_observed

        # SCM inference
        e = self._scm.batched_call(flatten_u, t, w_t, w_e)

        # Batched evidence (always cpu for dataloading)
        batched_evidence = self._batched_evidence_type(
            scm=self._scm,
            e_batched=e.detach().to('cpu'),
            t_batched=t.detach().to('cpu'),
            w_e_batched=w_e.detach().to('cpu'),
            w_t_batched=w_t.detach().to('cpu'),
            **self._evidence_kwargs,
        )

        # Returns
        if return_exogenous:
            return u.to('cpu'), BatchedEvidenceJoint(batched_evidence, j.detach().to('cpu'))
        return BatchedEvidenceJoint(batched_evidence, j.detach().to('cpu'))
