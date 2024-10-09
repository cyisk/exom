from torch.distributions import Normal

from common.scm import Equation, TensorSCM


@Equation(v='x1')
def f1(u1):
    return u1


@Equation(v='x2')
def f2(u2):
    return 2 - u2


@Equation(v='x3', pV=['x1', 'x2'])
def f3(x1, x2, u3):
    return 0.25 * x2 - 0.5 * x1 + 0.5 * u3


def collider_lin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='collider_lin',
        equations=[
            f1,
            f2,
            f3,
        ],
        exogenous_distrs={
            'u1': Normal(loc=0, scale=1),
            'u2': Normal(loc=0, scale=1),
            'u3': Normal(loc=0, scale=1),
        },
    )
