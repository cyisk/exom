import torch as th
from torch.distributions import Distribution
from typing import *

from model.counterfactual.jctf_estimator import *
from common import *


class NaiveSample(JointCounterfacutalEstimator):
    def __init__(self,
                 scm: TensorSCM,
                 # Sampling & Inference
                 max_len_joint: int = 1,
                 indicator: Callable | Dict[str, Callable] = l1_indicator(),
                 eval_sample_size: int = 1_000,
                 ) -> None:
        super().__init__(
            scm=scm,
            max_len_joint=max_len_joint,
            indicator=indicator,
            eval_sample_size=eval_sample_size,
        )
        self._exo_distr = SCMExogenousDistribution(scm=scm)
        self._dummy = th.nn.Linear(1, 1)

    def proposal(self,
                 evidence_batch: th.Tensor,
                 ) -> Distribution:
        w_j, _, _, _, _, _, _ = evidence_batch
        if not self._exo_distr.device == self.device:
            self._exo_distr = self._exo_distr.to(self.device)
        return self._exo_distr.expand(w_j.shape[:1])

    def configure_optimizers(self):  # Dummy optimizers
        return th.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):  # Dummy training
        return th.zeros(th.Size(), requires_grad=True)
