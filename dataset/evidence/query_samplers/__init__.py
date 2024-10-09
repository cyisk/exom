from typing import *

from dataset.evidence.evidence_sampler import EvidenceSampler, TensorSCM, Evidence
from dataset.evidence.batched_evidence_sampler import BatchedEvidenceSampler, BatchedEvidence
from dataset.evidence.query_samplers.base import *
from dataset.evidence.query_samplers.ate import *
from dataset.evidence.query_samplers.ett import *
from dataset.evidence.query_samplers.nde import *
from dataset.evidence.query_samplers.ctfde import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)
BatchedEvidenceLike = TypeVar('BatchedEvidenceLike',
                              bound=BatchedEvidence)


class QuerySampler(EvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 evidence_type: Type[EvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 query_type: str = 'ate',
                 **kwargs,
                 ) -> None:
        self._query_sampler: QuerySamplerCollection = {
            'ate': ATESamplerCollection,
            'ett': ETTSamplerCollection,
            'nde': NDESamplerCollection,
            'ctfde': CtfDESamplerCollection,
        }[query_type](scm=scm, **kwargs)

        super().__init__(
            scm=scm,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_hidden_sampler=self._query_sampler.joint_hidden_sampler(),
            joint_number_sampler=self._query_sampler.joint_number_sampler(),
            exogenous_sampler=self._query_sampler.exogenous_sampler(),
            antecedent_sampler=self._query_sampler.antecedent_sampler(),
            intervened_sampler=self._query_sampler.intervened_sampler(),
            observed_sampler=self._query_sampler.observed_sampler(),
            feature_observed_sampler=self._query_sampler.feature_observed_sampler(),
        )


class BatchedQuerySampler(BatchedEvidenceSampler):
    def __init__(self,
                 scm: TensorSCM,
                 batched_evidence_type: Type[BatchedEvidenceLike],
                 evidence_kwargs: Dict[str, Any] = {},
                 query_type: str = 'ate',
                 **kwargs,
                 ) -> None:
        self._query_sampler: QuerySamplerCollection = {
            'ate': ATESamplerCollection,
            'ett': ETTSamplerCollection,
            'nde': NDESamplerCollection,
            'ctfde': CtfDESamplerCollection,
        }[query_type](scm=scm, **kwargs)

        super().__init__(
            scm=scm,
            batched_evidence_type=batched_evidence_type,
            evidence_kwargs=evidence_kwargs,
            joint_hidden_sampler=self._query_sampler.joint_hidden_sampler(),
            joint_number_sampler=self._query_sampler.joint_number_sampler(),
            exogenous_sampler=self._query_sampler.exogenous_sampler(),
            antecedent_sampler=self._query_sampler.antecedent_sampler(),
            intervened_sampler=self._query_sampler.intervened_sampler(),
            observed_sampler=self._query_sampler.observed_sampler(),
            feature_observed_sampler=self._query_sampler.feature_observed_sampler(),
        )
