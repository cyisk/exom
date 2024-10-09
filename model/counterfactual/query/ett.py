from model.counterfactual.query.base import *

EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class ETT():
    def __new__(self,
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
                sample_size3: int = 10_000,
                ) -> None:
        q = Query(
            estimator=estimator,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
        )

        # ETT = E[Y_{X=x1}=y1 | X=x1] - E[Y_{X=x0}=y1 | X=x1]
        Y1 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x0})
        X = PotentialOutcome(Y=[X], y={X: x1}, X=[])
        log_PY1X = q.estimate_potential_outcomes(
            [Y1, X], sample_size1, as_log=True
        )  # P(Y_{X=x1}=y1, X=x1)
        log_PY2X = q.estimate_potential_outcomes(
            [Y2, X], sample_size2, as_log=True
        )  # P(Y_{X=x0}=y1, X=x1)
        log_PX = q.estimate_potential_outcomes(
            [X], sample_size3, as_log=True
        )  # P(X=x1)
        return math.exp(log_PY1X - log_PX) - math.exp(log_PY2X - log_PX)
