from model.counterfactual.query.base import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class ATE():
    def __new__(cls,
                estimator: JointCounterfacutalEstimator,
                Y: str,
                y1: th.Tensor,
                X: str,
                x0: th.Tensor,
                x1: th.Tensor,
                evidence_type: Type[EvidenceLike] = EvidenceContextConcat,
                evidence_kwargs: Optional[Dict[str, Any]] = {},
                sample_size1: int = 10_000,
                sample_size2: int = 10_000,
                ) -> None:
        q = Query(
            estimator=estimator,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
        )

        # ATE = E[Y_{X=x1}=y1] - E[Y_{X=x0}=y1]
        Y1 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x0})
        PY1 = q.estimate_potential_outcomes(
            [Y1], sample_size1
        )  # P(Y_{X=x1}=y1)
        PY2 = q.estimate_potential_outcomes(
            [Y2], sample_size2
        )  # P(Y_{X=x0}=y1)
        return PY1 - PY2
