import torch as th
from dataclasses import dataclass, field
from typing import *
from typing import Any

from common.scm import *
from dataset.evidence import *
from script.config import Config
from script.counterfactual.config_evidence import EvidenceConfig


@dataclass
class MCARBernoulliSamplerConfig(Config):
    joint_number_low: int = 1
    joint_number_high: int = 1
    prob_intervened: float = 0.2
    prob_observed: float = 0.8
    prob_feature_observed: float = 1.0

    def get_sampler(self,
                    scm: TensorSCM,
                    evidence_config: EvidenceConfig,
                    ) -> EvidenceSampler | BatchedEvidenceSampler:
        sampler_type = {
            True: BatchedMCARBernoulliSampler,
            False: MCARBernoulliSampler,
        }[evidence_config.batched]
        kwargs = {
            'scm': scm,
            'evidence_kwargs': evidence_config.get_evidence_kwargs(),
            'joint_number_low': self.joint_number_low,
            'joint_number_high': self.joint_number_high,
            'prob_intervened': self.prob_intervened,
            'prob_observed': self.prob_observed,
            'prob_feature_observed': self.prob_feature_observed,
        }
        if evidence_config.batched:
            kwargs['batched_evidence_type'] = evidence_config.get_evidence_type()
        else:
            kwargs['evidence_type'] = evidence_config.get_evidence_type()
        return sampler_type(**kwargs)


@dataclass
class QuerySamplerConfig(Config):
    query_type: str = 'ate'
    query_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in self.query_kwargs:
            if isinstance(self.query_kwargs[k], tuple):  # tensor
                self.query_kwargs[k] = th.tensor(self.query_kwargs[k])

    def get_sampler(self,
                    scm: TensorSCM,
                    evidence_config: EvidenceConfig,
                    ) -> EvidenceSampler | BatchedEvidenceSampler:
        sampler_type = {
            True: BatchedQuerySampler,
            False: QuerySampler,
        }[evidence_config.batched]
        kwargs = {
            'scm': scm,
            'evidence_kwargs': evidence_config.get_evidence_kwargs(),
            'query_type': self.query_type,
            **self.query_kwargs,
        }
        if evidence_config.batched:
            kwargs['batched_evidence_type'] = evidence_config.get_evidence_type()
        else:
            kwargs['evidence_type'] = evidence_config.get_evidence_type()
        return sampler_type(**kwargs)


@dataclass
class FixedSamplerConfig(Config):
    batch_path: str = ''

    def get_sampler(self,
                    scm: TensorSCM,
                    evidence_config: EvidenceConfig,
                    ) -> EvidenceSampler | BatchedEvidenceSampler:
        sampler_type = {
            True: BatchedFixedSampler,
            False: FixedSampler,
        }[evidence_config.batched]
        kwargs = {
            'scm': scm,
            'evidence_kwargs': evidence_config.get_evidence_kwargs(),
            'batches': self.batch_path,
        }
        if evidence_config.batched:
            kwargs['batched_evidence_type'] = evidence_config.get_evidence_type()
        else:
            kwargs['evidence_type'] = evidence_config.get_evidence_type()
        return sampler_type(**kwargs)


@dataclass
class SamplerConfig(Config):
    sampler_type: str
    sampler_kwargs: MCARBernoulliSamplerConfig

    def serialize(self) -> Dict[str, Any]:
        return {
            'sampler_type': self.sampler_type,
            'sampler_kwargs': self.sampler_kwargs.serialize(),
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        sampler_type = str(config_dict['sampler_type'])
        if sampler_type == 'mcar_bernoulli':
            sampler_kwargs = MCARBernoulliSamplerConfig.deserialize(
                config_dict=config_dict['sampler_kwargs']
            )
        elif sampler_type == 'query':
            sampler_kwargs = QuerySamplerConfig.deserialize(
                config_dict=config_dict['sampler_kwargs']
            )
        elif sampler_type == 'fixed':
            sampler_kwargs = FixedSamplerConfig.deserialize(
                config_dict=config_dict['sampler_kwargs']
            )
        else:
            raise ValueError("Unsupported Sampler type.")
        return SamplerConfig(
            sampler_type=sampler_type,
            sampler_kwargs=sampler_kwargs,
        )

    def get_sampler(self,
                    scm: TensorSCM,
                    evidence_config: EvidenceConfig,
                    ) -> EvidenceSampler | BatchedEvidenceSampler:
        return self.sampler_kwargs.get_sampler(scm, evidence_config)
