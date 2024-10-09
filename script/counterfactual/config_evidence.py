from dataclasses import dataclass, field
from typing import *
from typing import Any

from common.scm import *
from dataset.evidence import *
from script.config import Config

EvidenceLike = TypeVar('EvidenceLike',
                       bound=Evidence)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike',
                              bound=BatchedEvidence)


@dataclass
class EvidenceConfig(Config):
    evidence_type: str = 'context_concat'
    evidence_kwargs: dict = field(default_factory=dict)
    batched: bool = True
    max_len_joint: int = True

    def __post_init__(self):
        self.batched = bool(self.batched)

    def get_evidence_type(self) -> Type[Evidence] | Type[BatchedEvidence]:
        evidence_type = {
            'context_concat': {
                True: BatchedEvidenceContextConcat,
                False: EvidenceContextConcat,
            },
            'context_masked': {
                True: BatchedEvidenceContextMasked,
                False: EvidenceContextMasked,
            },
        }[self.evidence_type][self.batched]

        return evidence_type

    def get_evidence_kwargs(self) -> Dict[str, Any]:
        return self.evidence_kwargs
