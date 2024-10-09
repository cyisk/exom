import torch as th
from lightning import LightningModule
from math import prod
from torch.distributions import Distribution
from typing import *

from zuko.flows import *
from zuko.distributions import *

from common.graph.causal import *
from common.graph.causal import *
from common.scm import *
from model.zuko import *
from model.proxy_scm.causal_nf.metric import maximum_mean_discrepancy


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
        'maf': BMAF,
        'nsf': BNSF,
    }
    model_type = density_estimators[density_estimator_type]
    return model_type(**density_estimator_kwargs)


class CausalNormalizingFlow(LightningModule):
    def __init__(self,
                 # Structual Causal Model
                 causal_graph: DirectedMixedGraph,
                 endogenous_dimensions: Dict[str, th.Size],
                 # Standardization
                 prior_mean: th.Tensor = 0,
                 prior_std: th.Tensor = 1,
                 # Density estimator
                 density_estimator_type: str = 'maf',
                 density_estimator_kwargs: Optional[Dict[str, Any]] = {},
                 base_distribution_type: str = 'gaussian',
                 base_distribution_kwargs: Optional[Dict[str, Any]] = {},
                 # Learning
                 learning_rate: float = 1e-3,
                 ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._endo_dimensions = endogenous_dimensions
        self._endo_features = {
            v: prod(dim) for v, dim in endogenous_dimensions.items()
        }

        # For standardization
        if not isinstance(prior_mean, th.Tensor):
            prior_mean = th.tensor(prior_mean)
        self.register_buffer('_prior_mean', prior_mean)
        if not isinstance(prior_std, th.Tensor):
            prior_std = th.tensor(prior_std)
        self.register_buffer('_prior_std', prior_std)

        # Causal graph and dict graph
        assert causal_graph.is_dag  # Only work on dag
        self._graph = causal_graph.directed_graph

        # Get causal order
        self._endo_variables = topological_sort(self._graph)
        self._endo_orders = {v: i for i, v in enumerate(self._endo_variables)}

        # Make adjacency
        self._adjacency = th.zeros(
            (len(self._endo_variables), len(self._endo_variables))
        ).bool()
        for i, u in enumerate(self._endo_variables):
            for v in self._graph[u]:
                self._adjacency[self._endo_orders[v], i] = True
        repeats = th.tensor(list(self._endo_features.values()))
        self._adjacency = self._adjacency.repeat_interleave(
            repeats=repeats, dim=0
        )  # Repeat row for features
        self._adjacency = self._adjacency.repeat_interleave(
            repeats=repeats, dim=1
        )  # Repeat column for features

        # Initialize density estimators
        density_estimator_kwargs.update({
            'transforms': longest_path(self._graph),  # diameter is enough
            'features': sum(self._endo_features.values()),
            'context': 0,
        })
        self._density_estimator: BMAF = init_density_estimator(
            density_estimator_type=density_estimator_type,
            **density_estimator_kwargs
        )

        # Initialize base distribution
        base_distribution_kwargs.update({
            'features': sum(self._endo_features.values()),
        })
        self._base_distribution: LazyDistribution = init_base_distribution(
            base_distribution_type=base_distribution_type,
            **base_distribution_kwargs
        )
        if hasattr(self._density_estimator, 'base'):
            self._density_estimator.base = self._base_distribution

    def on_load_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        self._prior_mean = checkpoint['prior_mean']
        self._prior_std = checkpoint['prior_std']

    def on_save_checkpoint(self, checkpoint: th.Dict[str, th.Any]) -> None:
        checkpoint['prior_mean'] = self._prior_mean
        checkpoint['prior_std'] = self._prior_std

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
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

    def forward(self) -> Distribution:
        # Return observational distribution
        f_adjs = self._adjacency.to(self.device)
        return self._density_estimator(f_adjs=f_adjs)

    def training_step(self, batch, batch_idx):
        # Estimated observational distribution
        v = self.standardize(batch)
        p = self.forward()

        # Maximize likelihood
        loss1 = -p.log_prob(v)

        # Regularize
        jac = th.autograd.functional.jacobian(
            p.transform, v.mean(0), create_graph=True,
        )
        adj = self._adjacency.to(self.device).float()
        adj += th.eye(len(adj), device=self.device)  # Add diag
        loss2 = th.norm(jac[(adj == 0)], p=2)

        loss = (loss1 + loss2).mean()
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Estimated observational distribution
        v = self.standardize(batch)
        p = self.forward()

        # Samples
        x = p.sample((v.size(0), ))
        mmd = maximum_mean_discrepancy(v, x)

        self.log('mmd', mmd, on_step=False, on_epoch=True, prog_bar=True)
        return mmd

    def standardize(self, v: th.Tensor) -> th.Tensor:
        return (v - self._prior_mean) / self._prior_std

    def destandardize(self, v_overline: th.Tensor) -> th.Tensor:
        return v_overline * self._prior_std + self._prior_mean
