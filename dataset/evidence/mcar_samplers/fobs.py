import torch as th
from torch.distributions import Bernoulli
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class BernoulliFeatureObservedSampler:
    def __init__(self,
                 scm: TensorSCM,
                 prob: float = 0.5,
                 ) -> None:
        self._scm = scm
        self._prob = th.tensor(prob, device=scm.device)
        self._endo_dim = scm.endogenous_dimensions
        self._endo_features = th.tensor(
            list(self._scm.endogenous_features.values()),
            device=self._scm.device,
        )
        self._bernoulli = Bernoulli(probs=self._prob)

    def sample(self, observed: Set[str], h: None, j: None, U: TensorDict) -> TensorDict:
        return {
            v: self._bernoulli.sample(self._endo_dim[v]).bool()
            for v in self._endo_dim if v in observed
        }

    def batched_sample(self, observed: th.Tensor, h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.Tensor:
        w_observed = observed.repeat_interleave(self._endo_features, dim=1)
        return self._bernoulli.sample(w_observed.shape).bool() & w_observed
