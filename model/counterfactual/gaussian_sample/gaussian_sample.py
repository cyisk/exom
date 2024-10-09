import torch as th
from torch.distributions import Distribution
from typing import *

from zuko.distributions import DiagNormal

from model.counterfactual.jctf_estimator import *
from common import *


class GaussianSample(JointCounterfacutalEstimator):
    def __init__(self,
                 scm: TensorSCM,
                 # Sampling & Inference
                 max_len_joint: int = 1,
                 indicator: Callable | Dict[str, Callable] = l1_indicator(),
                 eval_sample_size: int = 1_000,
                 # Standardization
                 prior_u_mean: th.Tensor = 0,
                 prior_u_std: th.Tensor = 1,
                 ) -> None:
        super().__init__(
            scm=scm,
            max_len_joint=max_len_joint,
            indicator=indicator,
            eval_sample_size=eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
        )
        self._dummy = th.nn.Linear(1, 1)

    def proposal(self,
                 evidence_batch: th.Tensor,
                 ) -> Distribution:
        w_j, _, _, _, _, _, _ = evidence_batch
        return DiagNormal(th.zeros_like(self._prior_u_mean), th.ones_like(self._prior_u_mean)).expand(w_j.shape[:1])

    def configure_optimizers(self):  # Dummy optimizers
        return th.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):  # Dummy training
        return th.zeros(th.Size(), requires_grad=True)
