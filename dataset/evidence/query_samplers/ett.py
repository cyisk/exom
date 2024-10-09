from dataset.evidence.query_samplers.base import *


class ETTSamplerCollection(QuerySamplerCollection):
    def __init__(self,
                 scm: TensorSCM,
                 Y: str,
                 X: str,
                 x0: th.Tensor,
                 x1: th.Tensor,
                 ) -> None:
        super().__init__(scm)

        # ETT = E[Y_{X=x1}=y1 | X=x1] - E[Y_{X=x0}=y1 | X=x1]
        Y1 = PotentialOutcome(Y=[Y], X=[X], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], X=[X], x={X: x0})
        X = PotentialOutcome(Y=[X], X=[], x={})
        self.add_potential_outcomes(Y1, X)  # P(Y_{X=x1}, X)
        self.add_potential_outcomes(Y2, X)  # P(Y_{X=x0}, X)
        self.add_potential_outcomes(X)  # P(X)
