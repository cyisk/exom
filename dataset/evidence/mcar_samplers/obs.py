import torch as th
from torch.distributions import Bernoulli
from typing import *

from common.scm import *

TensorDict = Dict[str, th.Tensor]


class BernoulliObservedSampler:
    def __init__(self,
                 scm: TensorSCM,
                 prob: float = 0.5,
                 ensure_true: int = 1,
                 ) -> None:
        self._scm = scm
        self._prob = th.tensor(prob, device=scm.device)
        self._ensure_true = ensure_true
        self._endos = scm.endogenous_variables
        self._bernoulli = Bernoulli(probs=self._prob)

    def sample(self, h: None, j: None, U: TensorDict) -> Set[Any]:
        contained = self._bernoulli.sample((len(self._endos), )).bool()
        ensures = th.multinomial(th.ones(len(self._endos)),
                                 num_samples=self._ensure_true,
                                 replacement=False)
        contained[ensures] = True
        return set([self._endos[i] for i in range(len(self._endos)) if contained[i]])

    def batched_sample(self, h: th.Tensor, j: th.Tensor, u: th.Tensor) -> th.BoolTensor:
        contained = self._bernoulli.sample(
            (u.size(0), len(self._endos))
        ).bool()
        batch_idcs = th.arange(u.size(0))[:, None].expand(-1,
                                                          self._ensure_true)
        ensures = th.multinomial(th.ones(len(self._endos)).expand(u.size(0), -1),
                                 num_samples=self._ensure_true,
                                 replacement=False)
        contained[batch_idcs, ensures] = True
        return contained
