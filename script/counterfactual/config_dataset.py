from dataclasses import dataclass
from typing import *

from common.scm import *
from dataset.evidence import *
from script.config import Config
from script.counterfactual.config_evidence import EvidenceConfig
from script.counterfactual.config_sampler import SamplerConfig


@dataclass
class EvidenceDatasetConfig(Config):
    sampler: SamplerConfig
    size: int = 16384

    def get_dataset(self,
                    scm: TensorSCM,
                    evidence_config: EvidenceConfig,
                    ) -> EvidenceDataset | BatchedEvidenceDataset:
        sampler = self.sampler.get_sampler(scm, evidence_config)
        dataset_type = {
            True: BatchedEvidenceDataset,
            False: EvidenceDataset,
        }[evidence_config.batched]
        return dataset_type(
            scm=scm,
            sampler=sampler,
            size=self.size,
            max_len_joint=evidence_config.max_len_joint,
        )
