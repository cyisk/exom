import torch as th
from dataclasses import dataclass, field
from typing import *

from common.graph.causal import *
from common.scm import *
from model.proxy_scm.ncm import *
from script.config import Config


@dataclass
class GANNCMConfig(Config):
    # Density estimator
    exogenous_distribution_type: str = 'gaussian',
    ncm_hidden_features: list = [64, 64],
    critic_hidden_features: list = [64, 64],
    # Learning
    learning_rate: float = 1e-3
    n_critics: int = 5

    def get_model(self,
                  causal_graph: DirectedMixedGraph,
                  endogenous_dimensions: Dict[str, th.Size],
                  endogenous_logits: Dict[str, int],
                  exogenous_dimensions: Dict[str, th.Size],
                  ) -> GANNCM:
        return GANNCM(
            causal_graph=causal_graph,
            endogenous_dimensions=endogenous_dimensions,
            endogenous_logits=endogenous_logits,
            exogenous_dimensions=exogenous_dimensions,
            exogenous_distribution_type=self.exogenous_distribution_type,
            ncm_hidden_features=self.ncm_hidden_features,
            critic_hidden_features=self.critic_hidden_features,
            learning_rate=self.learning_rate,
            n_critics=self.n_critics,
        )

    def load_model(self,
                   causal_graph: DirectedMixedGraph,
                   endogenous_dimensions: Dict[str, th.Size],
                   endogenous_logits: Dict[str, int],
                   exogenous_dimensions: Dict[str, th.Size],
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> GANNCM:
        return GANNCM.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            causal_graph=causal_graph,
            endogenous_dimensions=endogenous_dimensions,
            endogenous_logits=endogenous_logits,
            exogenous_dimensions=exogenous_dimensions,
            exogenous_distribution_type=self.exogenous_distribution_type,
            ncm_hidden_features=self.ncm_hidden_features,
            critic_hidden_features=self.critic_hidden_features,
            learning_rate=self.learning_rate,
            n_critics=self.n_critics,
        )


@dataclass
class ModelConfig(Config):
    model_type: str
    model_kwargs: GANNCMConfig
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
        if model_type == 'gan_ncm':
            model_kwargs = GANNCMConfig.deserialize(
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
                  endogenous_logits: Dict[str, int],
                  exogenous_dimensions: Dict[str, th.Size],
                  ) -> GANNCM:
        return self.model_kwargs.get_model(
            causal_graph,
            endogenous_dimensions,
            endogenous_logits,
            exogenous_dimensions,
        )

    def load_model(self,
                   causal_graph: DirectedMixedGraph,
                   endogenous_dimensions: Dict[str, th.Size],
                   endogenous_logits: Dict[str, int],
                   exogenous_dimensions: Dict[str, th.Size],
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> GANNCM:
        return self.model_kwargs.load_model(
            causal_graph,
            endogenous_dimensions,
            endogenous_logits,
            exogenous_dimensions,
            checkpoint_path,
            map_location,
        )
