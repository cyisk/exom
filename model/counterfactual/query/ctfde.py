from model.counterfactual.query.base import *

TensorDict = Dict[str, th.Tensor]
EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class CtfDE():
    def __new__(self,
                estimator: JointCounterfacutalEstimator,
                Y: str,
                y1: th.Tensor,
                X: str,
                x0: th.Tensor,
                x1: th.Tensor,
                W: str,
                w: Distribution | List[TensorDict],
                evidence_type: Type[EvidenceLike] = EvidenceContextConcat,
                evidence_kwargs: Optional[Dict[str, Any]] = {},
                sample_size1: int = 10_000,
                sample_size2: int = 10_000,
                sample_size3: int = 10_000,
                marginal_sample_size: int = 100,
                ) -> None:
        q_kwargs = {}
        if isinstance(w, Distribution):
            q_kwargs['marginal_distrs'] = {W: w}
            q_kwargs['marginal_sample_size'] = marginal_sample_size
        else:
            q_kwargs['marginal_values'] = [{W: w_} for w_ in w]
            q_kwargs['marginal_weight'] = [0.5 for _ in w]
        q = Query(
            estimator=estimator,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
            **q_kwargs,
        )

        # CtfDE = \sum_w E[Y_{X=x1, W=w}=y1, W_{X=x0}=w | X=x1] - E[Y_{X=x0}=y1 | X=x1]
        # (Note: W is randomly sampled according to w_distr)
        Y1 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X, W], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x0})
        W = PotentialOutcome(Y=[W], X=[X], x={X: x0})
        X = PotentialOutcome(Y=[X], y={X: x1}, X=[])
        log_PY1WX = q.estimate_potential_outcomes(
            [Y1, W, X], sample_size1, as_log=True
        )  # P(Y_{X=x1, W=w}=y1, W_{X=x0}=w, X=x1
        log_PY2X = q.estimate_potential_outcomes(
            [Y2, X], sample_size2, as_log=True
        )  # P(Y_{X=x0}=y1, X=x1)
        log_PX = q.estimate_potential_outcomes(
            [X], sample_size3, as_log=True
        )  # P(X=x1)
        return math.exp(log_PY1WX - log_PX) - math.exp(log_PY2X - log_PX)
