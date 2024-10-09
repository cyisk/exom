import math
import torch as th
from torch.distributions import Distribution
from typing import *

from zuko.distributions import Joint

from common.scm import *
from dataset.utils import *
from dataset.evidence import Evidence, BatchedEvidence, EvidenceContextConcat, EvidenceJoint
from dataset.evidence.query_samplers.base import PotentialOutcome
from model.counterfactual.jctf_estimator import JointCounterfacutalEstimator

TensorDict = Dict[str, th.Tensor]
DistributionDict = Dict[str, Distribution]
EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class Query():
    def __init__(self,
                 estimator: JointCounterfacutalEstimator,
                 evidence_type: Type[EvidenceLike] = EvidenceContextConcat,
                 evidence_kwargs: Optional[Dict[str, Any]] = {},
                 marginal_distrs: Optional[DistributionDict] = None,
                 marginal_sample_size: Optional[DistributionDict] = None,
                 marginal_values: Optional[List[TensorDict]] = None,
                 marginal_weight: Optional[List[TensorDict]] = None,
                 ) -> None:
        self._estimator = estimator
        if issubclass(evidence_type, BatchedEvidence):
            self._evidence_type = evidence_type.unbatched_type()
        else:
            self._evidence_type = evidence_type
        self._evidence_kwargs = evidence_kwargs

        # For marginal sampling
        self._marginal_candidates = []
        assert marginal_distrs is None or marginal_values is None  # Either is None
        if marginal_distrs is not None and not len(marginal_distrs) == 0:
            self._marginal_candidates = set(marginal_distrs.keys())
            self._marginal_distrs = marginal_distrs
            self._marginal_sample_size = marginal_sample_size
        if marginal_values is not None and not len(marginal_values) == 0:
            for m in marginal_values:
                self._marginal_candidates += list(m.keys())
            self._marginal_candidates = set(self._marginal_candidates)
            self._marginal_values = [
                {x: to_float_tensor(x_val) for x, x_val in m.items()}
                for m in marginal_values
            ]
            weight_sum = sum(marginal_weight)
            self._marginal_weight = [
                weight / weight_sum for weight in marginal_weight
            ]

    def estimate_potential_outcomes(self,
                                    potential_outcomes: List[PotentialOutcome],
                                    sample_size: int,
                                    as_log: bool = False,
                                    ) -> float:
        # Check variables and values
        observed_to_sample = set()
        antecedent_to_sample = set()
        for potential_outcome in potential_outcomes:
            assert isinstance(potential_outcome, PotentialOutcome)
            # Check observed
            for Y in potential_outcome.Y:
                assert Y in self._estimator._scm.endogenous_variables
                # Randomly sampled observed variables
                if Y not in potential_outcome.y:
                    observed_to_sample.add(Y)
                    continue
                potential_outcome.y[Y] = to_float_tensor(
                    potential_outcome.y[Y]
                )
                assert self._estimator._scm.endogenous_dimensions[Y] == potential_outcome.y[Y].shape
            # Check intervened
            for X in potential_outcome.X:
                assert X in self._estimator._scm.endogenous_variables
                # Randomly sampled antecedent variables
                if X not in potential_outcome.x:
                    antecedent_to_sample.add(X)
                    continue
                potential_outcome.x[X] = to_float_tensor(
                    potential_outcome.x[X]
                )
                assert self._estimator._scm.endogenous_dimensions[X] == potential_outcome.x[X].shape
        # Check marginal is specified
        assert observed_to_sample.issubset(antecedent_to_sample) or\
            antecedent_to_sample.issubset(observed_to_sample), \
            "Must keep synchronization of variable values"
        assert observed_to_sample.issubset(self._marginal_candidates)
        assert antecedent_to_sample.issubset(self._marginal_candidates)
        marginal_variables = sorted(observed_to_sample | antecedent_to_sample)

        # Estimate an fully specific (no marginal) joint potential outcome
        def log_prob_of_potential_outcome(potential_outcomes: List[PotentialOutcome]):
            evidences = []
            for potential_outcome in potential_outcomes:
                evidence = self._evidence_type(
                    scm=self._estimator._scm,
                    observation={
                        Y: potential_outcome.y[Y]
                        for Y in potential_outcome.Y
                    },
                    intervention={
                        X: potential_outcome.x[X]
                        for X in potential_outcome.X
                    },
                    **self._evidence_kwargs,
                )
                evidences.append(evidence)
            joint_evidence = EvidenceJoint(evidences)
            p = self._estimator.proposal_distribution(joint_evidence)
            return self._estimator.estimate(p, joint_evidence, sample_size, True)

        # No marginal variable
        if len(marginal_variables) == 0:
            logprob = log_prob_of_potential_outcome(potential_outcomes)
            # print(potential_outcomes, math.exp(logprob))
            return logprob if as_log else math.exp(logprob)

        probs = []
        n = 0

        # Samples for marginal integral
        if hasattr(self, '_marginal_distrs'):
            marginal_distr = Joint(*[
                self._marginal_distrs[v] for v in marginal_variables
            ])
            marginal_dims = {
                v: self._estimator._scm.endogenous_dimensions[v] for v in marginal_variables
            }
            marginal_samples = marginal_distr.sample(
                (self._marginal_sample_size, )
            )
            marginal_logprob = marginal_distr.log_prob(marginal_samples)

            # For each marginal integral
            n = len(marginal_samples)
            for i, marginal_sample in enumerate(marginal_samples):
                for X, x_val in unbatch(marginal_sample, marginal_dims).items():
                    for k in range(len(potential_outcomes)):
                        if X in potential_outcomes[k].X:
                            potential_outcomes[k].x[X] = x_val
                        if X in potential_outcomes[k].Y:
                            potential_outcomes[k].y[X] = x_val
                logprob1 = log_prob_of_potential_outcome(potential_outcomes)
                logprob2 = marginal_logprob[i]
                probs.append(math.exp(logprob1 - logprob2))

        # Values for marginal integral
        if hasattr(self, '_marginal_values'):
            # For each marginal integral
            n = len(self._marginal_values)
            for i, marginal_sample in enumerate(self._marginal_values):
                for X, x_val in marginal_sample.items():
                    for k in range(len(potential_outcomes)):
                        if X in potential_outcomes[k].X:
                            potential_outcomes[k].x[X] = x_val
                        if X in potential_outcomes[k].Y:
                            potential_outcomes[k].y[X] = x_val
                logprob1 = log_prob_of_potential_outcome(potential_outcomes)
                logprob2 = math.log(self._marginal_weight[i])
                probs.append(math.exp(logprob1 - logprob2))
                # print(potential_outcomes, math.exp(logprob1 - logprob2))

        # Marginal prob
        return sum(probs) / n if not as_log \
            else math.log(sum(probs)) - math.log(n)
