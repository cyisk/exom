import torch as th
from typing import *

from torch.distributions import Distribution
from zuko.flows import *
from zuko.distributions import *

from dataset.synthetic import ObservationalDataset
from model.counterfactual.jctf_estimator import *
from model.counterfactual.exo_match.multi_context import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


def init_base_distribution(
    base_distribution_type: str = 'gaussian',
    **base_distribution_kwargs: Optional[Dict[str, Any]],
) -> th.nn.Module:
    base_distributions = {
        'gaussian': lambda features: Unconditional(
            DiagNormal,
            th.zeros(features),
            th.ones(features),
            buffer=True,
        ),
        'uniform': lambda features: Unconditional(
            BoxUniform,
            th.zeros(features),
            th.ones(features),
            buffer=True,
        ),
    }
    distr_type = base_distributions[base_distribution_type]
    return distr_type(**base_distribution_kwargs)


def init_density_estimator(
    density_estimator_type: str = 'maf',
    **density_estimator_kwargs,
) -> th.nn.Module:
    density_estimators = {
        'gmm': MultiContextGMM,
        'maf': MultiContextMAF,
        'nsf': MultiContextNSF,
        'ncsf': MultiContextNCSF,
        'nice': MultiContextNICE,
        'naf': MultiContextNAF,
        'unaf': MultiContextUNAF,
        'sospf': MultiContextSOSPF,
        'bpf': MultiContextBPF,
    }
    model_type = density_estimators[density_estimator_type]
    return model_type(**density_estimator_kwargs)


class ExogenousMatch(JointCounterfacutalEstimator):
    def __init__(self,
                 # Structual Causal Model
                 scm: TensorSCM,
                 # Evidence Type
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Optional[Dict[str, Any]] = {},
                 # Density estimator
                 density_estimator_type: str = 'maf',
                 density_estimator_kwargs: Optional[Dict[str, Any]] = {},
                 base_distribution_type: str = 'gaussian',
                 base_distribution_kwargs: Optional[Dict[str, Any]] = {},
                 # Learning & Sampling & Inference
                 learning_rate: float = 1e-3,
                 max_len_joint: int = 1,
                 indicator: Callable | Dict[str, Callable] = l1_indicator(),
                 eval_sample_size: int = 1_000,
                 # Standardization
                 prior_u_mean: th.Tensor = 0,
                 prior_u_std: th.Tensor = 1,
                 prior_update: bool = True,
                 prior_cold_starts: int = int(1e4),
                 prior_max_updates: int = 100,
                 ) -> None:
        super().__init__(
            scm=scm,
            max_len_joint=max_len_joint,
            indicator=indicator,
            eval_sample_size=eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
        )
        self._learning_rate = learning_rate
        self._evidence_type = evidence_type
        self._evidence_kwargs = evidence_kwargs

        # Initialize density estimators
        density_estimator_kwargs.update({
            'features': sum(scm.exogenous_features.values()),
            'context': evidence_type.context_features(scm, **evidence_kwargs),
            'max_context_num': max_len_joint,
        })
        self._density_estimator: LazyDistribution = init_density_estimator(
            density_estimator_type=density_estimator_type,
            **density_estimator_kwargs
        )

        # Initialize base distribution
        base_distribution_kwargs.update({
            'features': sum(scm.exogenous_features.values()),
        })
        self._base_distribution: LazyDistribution = init_base_distribution(
            base_distribution_type=base_distribution_type,
            **base_distribution_kwargs
        )
        if hasattr(self._density_estimator, 'base'):
            self._density_estimator.base = self._base_distribution

        # For endogenous standardization
        self._prior_update = prior_update
        self._prior_cold_starts = prior_cold_starts
        self._prior_max_updates = prior_max_updates
        if prior_cold_starts > 0:
            prior_ds = ObservationalDataset(scm, self._prior_cold_starts)
            self._prior_len = None
            self.register_buffer('_prior_v_mean', prior_ds.mean)
            self.register_buffer('_prior_v_std', prior_ds.std)
        else:
            prior_ds = ObservationalDataset(scm, 1)
            self.register_buffer('_prior_v_mean', th.zeros_like(prior_ds.mean))
            self.register_buffer('_prior_v_std', th.ones_like(prior_ds.std))

    def on_load_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._prior_v_mean = checkpoint['prior_v_mean']
        self._prior_v_std = checkpoint['prior_v_std']

    def on_save_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['prior_v_mean'] = self._prior_v_mean
        checkpoint['prior_v_std'] = self._prior_v_std

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-6,
        )
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min', factor=0.5, patience=5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss',
                'frequency': 1,
            }
        }

    def forward(self,
                context: th.Tensor,
                adjacency: th.BoolTensor = None,
                multi_context_mask: th.BoolTensor = None,
                ) -> Distribution:
        # Adjacency may be None
        adjacency = adjacency if isinstance(
            adjacency, th.Tensor
        ) else None

        # Multi Context Mask may be None
        multi_context_mask = multi_context_mask if isinstance(
            multi_context_mask, th.Tensor
        ) else None

        # Inference for proposal distribution
        return self._density_estimator(
            c=context,
            adj=adjacency,
            w_j=multi_context_mask,
        )

    def training_step(self, batch, batch_idx):
        u, w_j, e, w_e, t, w_t, context, adjacency = self.preprocess_batch(
            batch
        )

        # Prior update
        if self._prior_update:
            mask = feature_expand(w_j, th.Size((e.size(-1), )))
            self.update_prior(e, w_e & mask)
            self.update_prior(t, w_t & mask)

        # Estimate proposal distritbution
        context = self.standardize_context(context, w_e.bool(), w_t.bool())
        p = self.forward(context, adjacency, w_j)

        # Maximize likelihood
        logp = p.log_prob(u)
        # if density estimator yeilds -inf, skip
        mask = logp.isinf()
        if mask.any():
            return th.zeros(tuple(), device=self.device, requires_grad=True)
        loss = -logp.mean()

        self.log('loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def proposal(self,
                 evidence_batch: th.Tensor,
                 ) -> Distribution:
        w_j, e, w_e, t, w_t, context, adjacency = evidence_batch
        context = self.standardize_context(context, w_e.bool(), w_t.bool())
        p = self.forward(context, adjacency, w_j)
        return p

    @th.no_grad()
    def update_prior(self, v: th.Tensor, w_v: th.Tensor):
        if self._prior_max_updates >= 0 and self.current_epoch > self._prior_max_updates:
            return

        # Reshape for a flatten view
        v = v.reshape(-1, v.size(-1))
        v[v.isinf() | v.isinf()] = 0  # Avoid non-valid input
        w_v = w_v.reshape(-1, v.size(-1))
        l = w_v.sum(dim=0)

        # Mean and std
        mean_v = masked_mean(v, w_v, dim=0)
        std_v = masked_std(v, w_v, dim=0)

        # Updated mean and std
        if self._prior_len is None:
            self._prior_len = th.full_like(l, self._prior_cold_starts)
        l2 = self._prior_len + l
        sum2 = self._prior_v_mean * self._prior_len + mean_v * l
        mean2 = sum2 / l2
        varsum2 = (self._prior_v_std ** 2) * \
            self._prior_len + (std_v ** 2) * l
        std2 = th.sqrt(varsum2 / l2)
        self._prior_v_mean = mean2
        self._prior_v_std = std2
        self._prior_len = l2

    def standardize_context(self,
                            c: th.Tensor,
                            w_e: th.BoolTensor,
                            w_t: th.BoolTensor,
                            ) -> th.Tensor:
        return self._evidence_type.standardize(
            context=c,
            w_e=w_e,
            w_t=w_t,
            prior_mean=self._prior_v_mean,
            prior_std=self._prior_v_std,
            scm=self._scm,
            **self._evidence_kwargs
        )
