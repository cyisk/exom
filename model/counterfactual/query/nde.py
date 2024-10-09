from model.counterfactual.query.base import *

TensorDict = Dict[str, th.Tensor]
EvidenceLike = TypeVar('EvidenceLike', bound=Evidence)


class NDE():
    def __new__(self,
                estimator: JointCounterfacutalEstimator,
                Y: str,
                y1: th.Tensor,
                X: str,
                x0: th.Tensor,
                x1: th.Tensor,
                W: str,
                w: Distribution | List[th.Tensor],
                evidence_type: Type[EvidenceLike] = EvidenceContextConcat,
                evidence_kwargs: Optional[Dict[str, Any]] = {},
                sample_size1: int = 10_000,
                sample_size2: int = 10_000,
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

        # NDE = \sum_w E[Y_{X=x1, W=w}=y1, W_{X=x0}=w] - E[Y_{X=x0}=y1]
        # (Note: W is randomly sampled according to w_distr)
        Y1 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X, W], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], y={Y: y1}, X=[X], x={X: x0})
        W = PotentialOutcome(Y=[W], X=[X], x={X: x0})
        PY1W = q.estimate_potential_outcomes(
            [Y1, W], sample_size1
        )  # P(Y_{X=x1, W=w}=y1, W_{X=x0}=w)
        PY2 = q.estimate_potential_outcomes(
            [Y2], sample_size2
        )  # P(Y_{X=x0}=y1)
        return PY1W - PY2
