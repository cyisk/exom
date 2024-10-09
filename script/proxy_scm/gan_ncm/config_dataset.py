from dataclasses import dataclass, field
from typing import *

from common.scm import *
from dataset.synthetic import datasets
from dataset.synthetic import ObservationalDataset
from script.config import Config


@dataclass
class SyntheticSCMConfig(Config):
    name: str
    kwargs: dict = field(default_factory=dict)

    def get_scm(self) -> TensorSCM:
        kwargs = self.kwargs or {}
        return datasets[self.name](**kwargs)


@dataclass
class SCMConfig(Config):
    scm_type: str
    scm_kwargs: SyntheticSCMConfig

    def serialize(self) -> Dict[str, Any]:
        return {
            'scm_type': self.scm_type,
            'scm_kwargs': self.scm_kwargs.serialize(),
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        scm_type = str(config_dict['scm_type'])
        if scm_type == 'synthetic':
            scm_kwargs = SyntheticSCMConfig.deserialize(
                config_dict=config_dict['scm_kwargs']
            )
        elif scm_type == 'proxy':
            raise ValueError("Usage for proxy SCM has not been implemented.")
        else:
            raise ValueError("Unsupported SCM type.")
        return SCMConfig(
            scm_type=scm_type,
            scm_kwargs=scm_kwargs,
        )

    def get_scm(self) -> TensorSCM:
        return self.scm_kwargs.get_scm()


@dataclass
class ObservationalDatasetConfig(Config):
    size: int = 32768

    def get_dataset(self,
                    scm: TensorSCM,
                    ) -> ObservationalDataset:
        return ObservationalDataset(scm, self.size)
