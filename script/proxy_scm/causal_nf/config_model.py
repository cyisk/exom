import torch as th
from dataclasses import dataclass, field
from typing import *

from common.graph.causal import *
from common.scm import *
from model.proxy_scm.causal_nf import *
from script.config import Config


@dataclass
class CausalNFConfig(Config):
    # Density estimator
    density_estimator_type: str = 'maf'
    density_estimator_kwargs: dict = field(default_factory=dict)
    base_distribution_type: str = 'gaussian'
    base_distribution_kwargs: dict = field(default_factory=dict)
    # Learning
    learning_rate: float = 1e-3

    def get_model(self,
                  causal_graph: DirectedMixedGraph,
                  endogenous_dimensions: Dict[str, th.Size],
                  prior_mean: th.Tensor = 0,
                  prior_std: th.Tensor = 1,
                  ) -> CausalNormalizingFlow:
        return CausalNormalizingFlow(
            causal_graph=causal_graph,
            endogenous_dimensions=endogenous_dimensions,
            prior_mean=prior_mean,
            prior_std=prior_std,
            density_estimator_type=self.density_estimator_type,
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            learning_rate=self.learning_rate,
        )

    def load_model(self,
                   causal_graph: DirectedMixedGraph,
                   endogenous_dimensions: Dict[str, th.Size],
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> CausalNormalizingFlow:
        return CausalNormalizingFlow.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            causal_graph=causal_graph,
            endogenous_dimensions=endogenous_dimensions,
            density_estimator_type=self.density_estimator_type,
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            learning_rate=self.learning_rate,
        )


@dataclass
class ModelConfig(Config):
    model_type: str
    model_kwargs: CausalNFConfig
    # Checkpoint
    checkpoint_path: str = None

    def serialize(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs.serialize(),
            'checkpoint_path': self.checkpoint_path,
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        model_type = str(config_dict['model_type'])
        if model_type == 'causal_nf':
            model_kwargs = CausalNFConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        else:
            raise ValueError("Unsupported model type.")
        checkpoint_path = str(
            config_dict['checkpoint_path']
        ) if 'checkpoint_path' in config_dict else None
        return ModelConfig(
            model_type=model_type,
            model_kwargs=model_kwargs,
            checkpoint_path=checkpoint_path,
        )

    def get_model(self,
                  causal_graph: DirectedMixedGraph,
                  endogenous_dimensions: Dict[str, th.Size],
                  prior_mean: th.Tensor,
                  prior_std: th.Tensor,
                  ) -> CausalNormalizingFlow:
        return self.model_kwargs.get_model(causal_graph, endogenous_dimensions, prior_mean, prior_std)

    def load_model(self,
                   causal_graph: DirectedMixedGraph,
                   endogenous_dimensions: Dict[str, th.Size],
                   prior_mean: th.Tensor,
                   prior_std: th.Tensor,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> CausalNormalizingFlow:
        return self.model_kwargs.load_model(
            causal_graph, endogenous_dimensions, prior_mean, prior_std, checkpoint_path, map_location
        )
