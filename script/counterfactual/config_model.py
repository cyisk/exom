from dataclasses import dataclass, field
from typing import *

from common.scm import *
from model.counterfactual import *
from model.counterfactual.utils import *
from script.config import Config
from script.counterfactual.config_evidence import EvidenceConfig


def get_indicator(indicator_type: str, **indicator_kwargs):
    indicator = {
        'exact': exact_indicator,
        'l1': l1_indicator,
    }[indicator_type](**indicator_kwargs)
    return indicator


@dataclass
class NaiveSampleConfig(Config):
    # Sampling & Inference for ctf estimator
    indicator_type: str = 'l1'
    indicator_kwargs: dict = field(default_factory=dict)
    eval_sample_size: int = 1000

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  prior_update: bool = True,
                  ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return NaiveSample(
            scm=scm,
            indicator=indicator,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   prior_update: bool = True,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return NaiveSample.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            scm=scm,
            indicator=indicator,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
        )


@dataclass
class GaussianSampleConfig(Config):
    # Sampling & Inference for ctf estimator
    indicator_type: str = 'l1'
    indicator_kwargs: dict = field(default_factory=dict)
    eval_sample_size: int = 1000

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  prior_update: bool = True,
                  ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return GaussianSample(
            scm=scm,
            indicator=indicator,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   prior_update: bool = True,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return GaussianSample.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            scm=scm,
            indicator=indicator,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )


@dataclass
class ExoMatchConfig(Config):
    # Density estimator
    density_estimator_type: str = 'maf'
    density_estimator_kwargs: dict = field(default_factory=dict)
    base_distribution_type: str = 'gaussian'
    base_distribution_kwargs: dict = field(default_factory=dict)
    # Sampling & Inference for ctf estimator
    indicator_type: str = 'l1'
    indicator_kwargs: dict = field(default_factory=dict)
    eval_sample_size: int = 1000
    # Learning
    learning_rate: float = 1e-3

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  prior_update: bool = True,
                  ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return ExogenousMatch(
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_type=self.density_estimator_type,
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   prior_update: bool = True,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> ExogenousMatch:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return ExogenousMatch.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_type=self.density_estimator_type,
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )


@dataclass
class NeuralConfig(Config):
    # Density estimator
    density_estimator_kwargs: dict = field(default_factory=dict)
    base_distribution_type: str = 'gaussian'
    base_distribution_kwargs: dict = field(default_factory=dict)
    # Sampling & Inference for ctf estimator
    indicator_type: str = 'l1'
    indicator_kwargs: dict = field(default_factory=dict)
    eval_sample_size: int = 1000
    cold_starts: int = 0
    train_sample_size: int = 1000
    # Learning
    learning_rate: float = 1e-3

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  prior_update: bool = True,
                  ) -> NeuralIS:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return NeuralIS(
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            cold_starts=self.cold_starts,
            train_sample_size=self.train_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   prior_update: bool = True,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> NeuralIS:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return NeuralIS.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_kwargs=self.density_estimator_kwargs,
            base_distribution_type=self.base_distribution_type,
            base_distribution_kwargs=self.base_distribution_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            cold_starts=self.cold_starts,
            train_sample_size=self.train_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )


@dataclass
class CrossEntropyConfig(Config):
    # Density estimator
    density_estimator_kwargs: dict = field(default_factory=dict)
    # Sampling & Inference for ctf estimator
    indicator_type: str = 'l1'
    indicator_kwargs: dict = field(default_factory=dict)
    eval_sample_size: int = 1000
    cold_starts: int = 0
    train_sample_size: int = 1000
    # Learning
    learning_rate: float = 1e-3

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  prior_update: bool = True,
                  ) -> CrossEntropyIS:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return CrossEntropyIS(
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_kwargs=self.density_estimator_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            cold_starts=self.cold_starts,
            train_sample_size=self.train_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   prior_update: bool = True,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> CrossEntropyIS:
        indicator = get_indicator(
            self.indicator_type, **self.indicator_kwargs
        )
        return CrossEntropyIS.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            scm=scm,
            evidence_type=evidence_config.get_evidence_type(),
            evidence_kwargs=evidence_config.get_evidence_kwargs(),
            density_estimator_kwargs=self.density_estimator_kwargs,
            indicator=indicator,
            learning_rate=self.learning_rate,
            max_len_joint=evidence_config.max_len_joint,
            eval_sample_size=self.eval_sample_size,
            cold_starts=self.cold_starts,
            train_sample_size=self.train_sample_size,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=prior_update,
        )


@dataclass
class ModelConfig(Config):
    model_type: str
    model_kwargs: NaiveSampleConfig | ExoMatchConfig
    # Prior update
    prior_update: bool = True
    # Checkpoint
    checkpoint_path: str = None

    def __post_init__(self):
        self.prior_update = bool(self.prior_update)

    def serialize(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs.serialize(),
            'prior_update': 'true' if self.prior_update else 'false',
            'checkpoint_path': self.checkpoint_path,
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        model_type = str(config_dict['model_type'])
        if model_type == 'naive':
            model_kwargs = NaiveSampleConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        elif model_type == 'gaussian':
            model_kwargs = GaussianSampleConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        elif model_type == 'exom':
            model_kwargs = ExoMatchConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        elif model_type == 'nis':
            model_kwargs = NeuralConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        elif model_type == 'ce':
            model_kwargs = CrossEntropyConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        else:
            raise ValueError("Unsupported model type.")
        checkpoint_path = str(
            config_dict['checkpoint_path']
        ) if 'checkpoint_path' in config_dict else None
        prior_update = bool(
            config_dict['prior_update']) if 'prior_update' in config_dict else False
        return ModelConfig(
            model_type=model_type,
            model_kwargs=model_kwargs,
            prior_update=prior_update,
            checkpoint_path=checkpoint_path,
        )

    def get_model(self,
                  scm: TensorSCM,
                  evidence_config: EvidenceConfig,
                  prior_u_mean: th.Tensor,
                  prior_u_std: th.Tensor,
                  ) -> Any:
        return self.model_kwargs.get_model(
            scm=scm,
            evidence_config=evidence_config,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=self.prior_update,
        )

    def load_model(self,
                   scm: TensorSCM,
                   evidence_config: EvidenceConfig,
                   prior_u_mean: th.Tensor,
                   prior_u_std: th.Tensor,
                   checkpoint_path: str = None,
                   map_location: str = None,
                   ) -> Any:
        return self.model_kwargs.load_model(
            scm=scm,
            evidence_config=evidence_config,
            prior_u_mean=prior_u_mean,
            prior_u_std=prior_u_std,
            prior_update=self.prior_update,
            checkpoint_path=checkpoint_path,
            map_location=map_location,
        )
