from dataset.evidence.query_samplers.base import *


class ATESamplerCollection(QuerySamplerCollection):
    def __init__(self,
                 scm: TensorSCM,
                 Y: str,
                 X: str,
                 x0: th.Tensor,
                 x1: th.Tensor,
                 ) -> None:
        super().__init__(scm)

        # ATE = E[Y_{X=x1}=y1] - E[Y_{X=x0}=y1]
        Y1 = PotentialOutcome(Y=[Y], X=[X], x={X: x1})
        Y2 = PotentialOutcome(Y=[Y], X=[X], x={X: x0})
        self.add_potential_outcomes(Y1)  # P(Y_{X=x1})
        self.add_potential_outcomes(Y2)  # P(Y_{X=x0})
