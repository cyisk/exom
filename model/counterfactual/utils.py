import abc
import copy
import torch as th
from typing import *

from dataset.utils import *
from torch.distributions import Distribution, Transform, constraints
from zuko.distributions import Joint
from common.scm import *

TensorDict = Dict[str, th.Tensor]

"""Distribution"""


class SCMExogenousDistribution(Joint):
    def __init__(self, scm: TensorSCM) -> None:
        self._scm = copy.deepcopy(scm)  # Not change original scm
        if isinstance(scm.exogenous_distributions, dict):
            super().__init__(*[
                scm.exogenous_distributions[u] for u in scm.exogenous_variables
            ])
        else:
            super().__init__(*[scm.exogenous_distributions])

    def to(self, device: Optional[str | th.device | int] = None) -> "SCMExogenousDistribution":
        return SCMExogenousDistribution(self._scm.to(device))

    def expand(self, batch_shape: th.Size, new: Distribution = None) -> Distribution:
        new = self._get_checked_instance(SCMExogenousDistribution, new)
        new = super().expand(batch_shape, new)
        new._scm = self._scm
        return new

    @property
    def device(self) -> str | th.device | int:
        return self._scm.device


"""Standardization"""


class StandardTransform(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: th.Tensor,
        scale: th.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shift = shift
        self.scale = scale

    def _call(self, x: th.Tensor) -> th.Tensor:
        return (x - self.shift) / self.scale

    def _inverse(self, y: th.Tensor) -> th.Tensor:
        return y * self.scale + self.shift

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return -self.scale.abs().log().expand(x.shape)

    def __str__(self):
        my_str = "StandardTransform(shift={}, scale={})".format(
            self.shift, self.scale)
        return my_str

    def to(self, device: Any) -> "StandardTransform":
        return StandardTransform(self.shift.to(device=device), self.scale.to(device=device))


"""Evaluation"""


class MaskedEvaluation(abc.ABC):
    @abc.abstractmethod
    def eval(self, x_hat: th.Tensor, x: th.Tensor, mask: th.Tensor, w_j: th.Tensor = None) -> th.Tensor:
        raise NotImplementedError


class MaskedEndoEvaluation(MaskedEvaluation):
    def __init__(self,
                 scm: TensorSCM,
                 eval_func: Callable | Dict[str, Callable],
                 ) -> None:
        self._scm = scm
        self._dims = scm.endogenous_dimensions
        self._eval_func = eval_func

    def __repr__(self) -> str:
        return f'masked_endo_eval[{self._eval_func.__repr__()}]'

    def variable_eval(self,
                      X_hat: TensorDict,
                      X: TensorDict,
                      Mask: TensorDict,
                      ) -> Tuple[TensorDict, Dict[str, th.Size]]:
        assert set(X_hat.keys()) == set(X.keys()) == set(self._dims.keys())
        batch_shape = batchshape(X_hat, self._dims)
        Eval = {}
        dim_Eval = {}
        for v, v_dim in self._dims.items():
            # For convenience, flatten batch shape
            V_hat = X_hat[v].reshape(-1, *v_dim)
            V = X[v].reshape(-1, *v_dim)
            V_mask = Mask[v].reshape(-1, *v_dim)
            # Call evaluation function
            if isinstance(self._eval_func, dict):
                evald = self._eval_func[v](V_hat, V, V_mask)
            else:
                evald = self._eval_func(V_hat, V, V_mask)
            # Unflatten batch shape
            evald = evald.reshape(*batch_shape, *evald.shape[1:])
            # Record evaluated result and shape
            Eval[v] = evald
            dim_Eval[v] = evald.shape[len(batch_shape):]
        return Eval, dim_Eval

    def event_eval(self,
                   x_hat: th.Tensor,
                   x: th.Tensor,
                   variable_mask: th.Tensor,
                   joint_mask: th.Tensor,
                   ) -> Tuple[TensorDict, th.Size]:
        eval = self._eval_func(x_hat, x, variable_mask, joint_mask)
        dim = eval.shape[joint_mask.dim():]
        return eval, dim

    def eval(self,
             x_hat: th.Tensor,
             x: th.Tensor,
             variable_mask: th.Tensor,
             joint_mask: th.Tensor,
             ) -> Tuple[th.Tensor, Any]:
        if self._eval_func.granularity == 'variable':
            Eval, dim_Eval = self.variable_eval(
                X_hat=unbatch(x_hat, self._dims),
                X=unbatch(x, self._dims),
                Mask=unbatch(variable_mask, self._dims),
            )
            # Returned shape is [batch_size, endo_size, ...]
            return feature_masked_scatter(batch(Eval, dim_Eval), joint_mask, default=1), dim_Eval
        elif self._eval_func.granularity == 'joint':
            x_hat = feature_masked_scatter(x_hat, joint_mask)
            x = feature_masked_scatter(x, joint_mask)
            variable_mask = feature_masked_scatter(variable_mask, joint_mask)
            return self.event_eval(x_hat, x, variable_mask, joint_mask)
        else:
            raise TypeError("Unknown granularity")


"""Indicator"""


class l1_indicator:
    def __init__(self, epsilon: float = 1e-2):
        self.epsilon = epsilon
        self.granularity = 'variable'

    def __repr__(self) -> str:
        return f'l1_indicator[eps={self.epsilon}]'

    def __call__(self, v_hat, v, mask) -> th.BoolTensor:
        dist = th.abs(v - v_hat)
        # If mask == False (unobserved), it implies valid
        indicate = (dist < self.epsilon) | (~mask)
        feat_dims = (th.arange(mask.dim()-1)+1).tolist()
        return th.all(indicate, dim=feat_dims)


class exact_indicator(l1_indicator):
    def __init__(self):
        super().__init__(epsilon=1e-8)

    def __repr__(self) -> str:
        return 'exact_indicator'


class range_indicator:
    def __init__(self, lb: th.Tensor, ub: th.Tensor):
        assert lb.shape == ub.shape
        self.lb = lb
        self.ub = ub
        self.granularity = 'joint'

    def __repr__(self) -> str:
        return f'range_indicator[[{str(self.lb.tolist())}],[{str(self.ub.tolist())}]]'

    def __call__(self, v_hat, v, variable_mask, joint_mask) -> th.BoolTensor:
        assert (self.lb.shape == v_hat.shape[-self.lb.dim():])
        # If mask == False (unobserved), it implies valid
        indicate = ((v_hat >= self.lb.to(v_hat.device))
                    & (v_hat <= self.ub.to(v_hat.device))) | (~variable_mask)
        feat_dims = (th.arange(joint_mask.dim(),
                     variable_mask.dim())).tolist()
        return th.all(indicate, dim=feat_dims) | (~joint_mask)


class point_indicator(range_indicator):
    def __init__(self, x: th.Tensor):
        super().__init__(lb=x-1e-8, ub=x+1e-8)

    def __repr__(self) -> str:
        return 'point_indicator'
