import torch as th
from torch.distributions import Distribution
from dataclasses import dataclass, field
from typing import *

from zuko.distributions import Joint

from common.scm import *
from dataset.utils import *
from dataset.evidence.mcar_samplers.fobs import *


@dataclass
class QuerySampler(object):
    sample: Callable
    batched_sample: Callable


@dataclass
class PotentialOutcome:
    # Y_{X=x}=y
    Y: List[str] = field(default_factory=list)
    X: List[str] = field(default_factory=list)
    y: Dict[str, Any] = field(default_factory=dict)
    x: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"[{','.join([f'{y}={self.y[y]}' for y in self.Y])}]" +\
            f"_[{','.join([f'{x}={self.x[x]}' for x in self.X])}]"


TensorDict = Dict[str, th.Tensor]
DistributionDict = Dict[str, Distribution]


class QuerySamplerCollection():
    def __init__(self,
                 scm: TensorSCM,
                 marginal_distrs: Optional[DistributionDict] = None,
                 marginal_values: Optional[List[TensorDict]] = None,
                 ) -> None:
        self._scm = scm
        self._potentials: Iterable[Iterable[PotentialOutcome]] = []

        # For marginal sampling
        self._marginal_candidates = []
        assert marginal_distrs is None or marginal_values is None  # Either is None
        if marginal_distrs is not None and not len(marginal_distrs) == 0:
            self._marginal_candidates = set(sorted(marginal_distrs.keys()))
            self._marginal_distrs = marginal_distrs
            self._joint_marginal_distr = Joint(*[
                self._marginal_distrs[v] for v in self._marginal_candidates
            ])
            self._marginal_dims = {
                v: self._scm.endogenous_dimensions[v]
                for v in self._marginal_candidates
            }
        if marginal_values is not None and not len(marginal_values) == 0:
            for m in marginal_values:
                self._marginal_candidates += list(m.keys())
            self._marginal_candidates = set(sorted(self._marginal_candidates))
            self._marginal_values = [
                {x: to_float_tensor(x_val) for x, x_val in m.items()}
                for m in marginal_values
            ]

    def add_potential_outcomes(self, *potential_outcomes) -> None:
        # Check valid, parse to tensor, check shape
        for potential in potential_outcomes:
            assert isinstance(potential, PotentialOutcome)
            for Y in potential.Y:
                assert Y in self._scm.endogenous_variables
                # Observed value is not required in sampler
                # assert self._scm.endogenous_dimensions[Y] == potential.y[Y].shape
            for X in potential.X:
                assert X in self._scm.endogenous_variables
                if X not in potential.x:  # Randomly sampled antecedent variables
                    continue
                potential.x[X] = to_float_tensor(
                    potential.x[X]
                ).to(self._scm.device)
                assert self._scm.endogenous_dimensions[X] == potential.x[X].shape
        # Append potential outcome
        self._potentials.append(list(potential_outcomes))

    def joint_hidden_sampler(self) -> QuerySampler:
        def sample() -> int:
            # Choose one of potential outcomes
            return th.randint(0, len(self._potentials), device=self._scm.device).float()

        def batched_sample(batch_size: int) -> th.Tensor:
            # Choose one of potential outcomes for each batch
            return th.randint(0, len(self._potentials), size=(batch_size, ), device=self._scm.device).float()

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def joint_number_sampler(self) -> QuerySampler:
        def sample(h: int) -> int:
            # Just equal to h-th potential outcomes' length
            return len(self._potentials[h])

        def batched_sample(batch_size: int, h: th.Tensor) -> th.Tensor:
            # Just equal to h-th potential outcomes' length for each batch
            j = th.zeros((batch_size, ), device=self._scm.device).int()
            for i in range(len(self._potentials)):
                j[h == i] = len(self._potentials[i])
            return j

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def exogenous_sampler(self) -> QuerySampler:
        def sample(h: None) -> Any:
            # From scm exogenous distribution
            return self._scm.sample()

        def batched_sample(batch_size: int, h: th.Tensor) -> th.Tensor:
            # From scm exogenous distribution
            U = self._scm.sample(batch_size)
            return batch(U, self._scm.exogenous_dimensions)

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def antecedent_sampler(self) -> QuerySampler:
        def sample(h: int, j: int, U: TensorDict) -> Any:
            U_ = self._scm.sample()
            S_ = self._scm(U_)
            # Intervene according to potential outcome
            potential: PotentialOutcome = self._potentials[h][j]
            for X in potential.X:
                S_[X] = potential.x[X]
            # Samples for marginal antecedent
            if hasattr(self, '_marginal_distrs'):
                marginal_sample = self._joint_marginal_distr.sample()
                M_ = unbatch(marginal_sample, self._marginal_dims)
                for X in M_:
                    S_[X] = M_[X]
            # Values for marginal antecedent
            if hasattr(self, '_marginal_values'):
                for X in self._marginal_values:
                    S_[X] = M_[X]
            return S_

        def batched_sample(h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
            U_ = self._scm.sample(u.size(0))
            S_ = self._scm(U_)
            batch_shape = th.Size((u.size(0), ))
            # Intervene according to potential outcome for each batch
            for i in range(len(self._potentials)):
                for k in range(len(self._potentials[i])):
                    potential: PotentialOutcome = self._potentials[i][k]
                    for X in potential.X:
                        if X not in potential.x:  # Randomly sampled antecedent variables
                            continue
                        # Expand shape for broadcast
                        h_ = feature_expand(h, potential.x[X].shape[h.dim():])
                        j_ = feature_expand(j, potential.x[X].shape[j.dim():])
                        x_ = batch_expand(potential.x[X], batch_shape)
                        S_[X][(h_ == i) & (j_ == k)] = \
                            x_[(h_ == i) & (j_ == k)]
            return batch(S_, self._scm.endogenous_dimensions)

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def intervened_sampler(self) -> QuerySampler:
        def sample(h: int, j: int, U: TensorDict) -> Set[Any]:
            # Intervene according to potential outcome
            potential: PotentialOutcome = self._potentials[h][j]
            return set(potential.X)

        def batched_sample(h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
            mask_shape = (u.size(0), len(self._scm.endogenous_variables))
            contained = th.zeros(mask_shape, device=self._scm.device).bool()
            # Intervene according to potential outcome for each batch
            for i in range(len(self._potentials)):
                for k in range(len(self._potentials[i])):
                    potential: PotentialOutcome = self._potentials[i][k]
                    for t, X in enumerate(self._scm.endogenous_variables):
                        if X in potential.X:
                            contained[:, t][(h == i) & (j == k)] = True
            return contained

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def observed_sampler(self) -> QuerySampler:
        def sample(h: int, j: int, U: TensorDict) -> Set[Any]:
            # Observe according to potential outcome
            potential: PotentialOutcome = self._potentials[h][j]
            return set(potential.X)

        def batched_sample(h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
            mask_shape = (u.size(0), len(self._scm.endogenous_variables))
            contained = th.zeros(mask_shape, device=self._scm.device).bool()
            # Observe according to potential outcome for each batch
            for i in range(len(self._potentials)):
                for k in range(len(self._potentials[i])):
                    potential: PotentialOutcome = self._potentials[i][k]
                    for t, Y in enumerate(self._scm.endogenous_variables):
                        if Y in potential.Y:
                            contained[:, t][(h == i) & (j == k)] = True
            return contained

        return QuerySampler(sample=sample, batched_sample=batched_sample)

    def feature_observed_sampler(self) -> QuerySampler:
        return BernoulliFeatureObservedSampler(self._scm, 1.)
