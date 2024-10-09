import copy
import torch as th
from typing import *

from common.scm import *
from dataset.evidence.evidence import *
from dataset.evidence.evidence_joint import *

TensorDict = Dict[str, th.Tensor]


class _TensorDictSampler:
    def __init__(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs) -> TensorDict:
        pass


class _RangeSampler:
    def __init__(self, low: int, high: int, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs) -> int:
        pass


class _SetSampler:
    def __init__(self, universal: Set[Any], *args, **kwargs):
        pass

    def sample(self, *args, **kwargs) -> Set[Any]:
        pass


RangeSampler = TypeVar('RangeSampler', bound=_RangeSampler)
TensorDictSampler = TypeVar('TensorDictSampler', bound=_TensorDictSampler)
SetSampler = TypeVar('SetSampler', bound=_SetSampler)
EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class EvidenceSampler:
    def __init__(self,
                 scm: TensorSCM,
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Dict[str, Any],
                 joint_hidden_sampler: TensorDictSampler,
                 joint_number_sampler: RangeSampler,
                 exogenous_sampler: TensorDictSampler,
                 antecedent_sampler: TensorDictSampler,
                 intervened_sampler: SetSampler,
                 observed_sampler: SetSampler,
                 feature_observed_sampler: TensorDictSampler,
                 ) -> None:
        self._scm = scm
        self._evidence_type = evidence_type
        self._evidence_kwargs = evidence_kwargs
        self._j_sampler = joint_number_sampler
        self._h_sampler = joint_hidden_sampler
        self._u_sampler = exogenous_sampler
        self._s_sampler = antecedent_sampler
        self._itv_sampler = intervened_sampler
        self._obs_sampler = observed_sampler
        self._fobs_sampler = feature_observed_sampler

    def sample(self,
               return_exogenous: bool = False,
               ) -> EvidenceJoint:
        # Sample joint inference hidden state
        h = self._h_sampler.sample()

        # Sample joint inference size
        j = self._j_sampler.sample(h)

        # Sample exogenous
        U = self._u_sampler.sample(h)

        evidences = []
        for k in range(j):  # Independent joint sampling
            # Sample missing variables
            intervened = self._itv_sampler.sample(h, k, U)
            observed = self._obs_sampler.sample(h, k, U)
            observed |= intervened  # intervened is also observed

            # Sample antecedent variables
            S = self._s_sampler.sample(h, k, U)

            # Sample observed features
            feature_observed = self._fobs_sampler.sample(observed, h, k, U)

            # W_T
            W_T = pad_variable_mask(intervened,
                                    self._scm.endogenous_dimensions)

            # W_E
            W_E = pad_variable_mask(observed,
                                    self._scm.endogenous_dimensions)
            # intervened is also observed
            W_E = {v: W_E[v] | W_T[v] for v in W_E}
            W_E = {
                v: W_E[v] & feature_observed[v].to(W_E[v].device)
                if v in feature_observed
                else W_E[v] for v in W_E
            }

            # T = S \otimes W_E
            T = S
            for v in S:
                T[v][~W_T[v]] = 0

            # E = scm(U, T, W_T, W_E)
            W_E = {v: W_E[v].to(self._scm.device).detach() for v in W_E}
            W_T = {v: W_T[v].to(self._scm.device).detach() for v in W_T}
            E = self._scm(U, T, W_T, W_E)

            # TensorDict with missing variables (always cpu for dataloading)
            observation = {v: E[v].detach().to('cpu') for v in observed}
            intervention = {v: T[v].detach().to('cpu') for v in intervened}
            feature_observed = {
                v: feature_observed[v].detach().to('cpu') for v in observed
            }

            # Evidence, allowing custom evidence type
            evidence = self._evidence_type(
                scm=self._scm,
                observation=observation,
                intervention=intervention,
                feature_observed=feature_observed,
                **self._evidence_kwargs,
            )

            evidences.append(evidence)

        if return_exogenous:
            return U, EvidenceJoint(evidences)
        return EvidenceJoint(evidences)
