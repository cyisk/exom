import torch as th
from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='w')
def f1(uw, uxw, uyw):
    return uw + uxw + th.tanh(uyw)


@Equation(v='z', pV='w')
def f2(w, uz):
    return w + 0.5 * uz


@Equation(v='x', pV='z')
def f3(z, ux, uxw):
    return z ** 3 / 15 + th.exp(ux / 2) + uxw / 4


@Equation(v='y', pV='x')
def f4(x, uy, uyw):
    return x - uy * 0.2 + uyw * 0.5


def napkin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='napkin',
        equations=[
            f1,
            f2,
            f3,
            f4,
        ],
        exogenous_distrs={
            'uw': Normal(loc=0, scale=1),
            'uz': Normal(loc=0, scale=1),
            'ux': Normal(loc=0, scale=1),
            'uy': Normal(loc=0, scale=1),
            'uxw': Normal(loc=0, scale=1),
            'uyw': Normal(loc=0, scale=1),
        },
    )
