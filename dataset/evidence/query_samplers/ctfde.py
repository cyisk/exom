from dataset.evidence.query_samplers.base import *


class CtfDESamplerCollection(QuerySamplerCollection):
    def __init__(self,
                 scm: TensorSCM,
                 Y: str,
                 X: str,
                 W: str,
                 x0: th.Tensor,
                 x1: th.Tensor,
                 w: Distribution | List[th.Tensor],
                 ) -> None:
        q_kwargs = {}
        if isinstance(w, Distribution):
            q_kwargs['marginal_distrs'] = {W: w}
        else:
            q_kwargs['marginal_values'] = [{W: w_} for w_ in w]
        super().__init__(
            scm=scm,
            **q_kwargs
        )

        # CtfDE = \sum_w E[Y_{X=x1, W=w}=y1, W_{X=x0}=w | X=x1] - E[Y_{X=x0}=y1 | X=x1]
        # (Note: W is randomly sampled)
        Y1 = PotentialOutcome(Y=[Y], X=[X, W], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], X=[X], x={X: x0})
        W = PotentialOutcome(Y=[W], X=[X], x={X: x0})
        X = PotentialOutcome(Y=[X], X=[], x={})
        self.add_potential_outcomes(Y1, W, X)  # P(Y_{X=x1, W=w}, W_{X=x0}, X)
        self.add_potential_outcomes(Y2, X)  # P(Y_{X=x0}, X)
        self.add_potential_outcomes(X)  # P(Y_{X=x0}, X)
