import math
import torch as th
from torch.distributions import Uniform

from common.scm import Equation, TensorSCM


def s(x):
    if isinstance(x, th.Tensor):
        return th.log(1 + th.exp(x))
    else:
        return math.log(1 + math.log(x))


def L(x, y):
    return s(x + 1) + s(0.5 + y) - 3


def cdf_1(mu, beta, x):
    return mu - beta * th.sign(x - 0.5) * math.sqrt(2) * th.erfinv(2 * th.abs(x - 0.5))


@Equation(v='x1')
def f1(u1):
    return s(1.8 * u1) - 1


@Equation(v='x2', pV='x1')
def f2(x1, u2):
    return 0.25 * u2 + L(x1, 0) * 1.5


@Equation(v='x3', pV='x1')
def f3(x1, u3):
    return L(x1, u3)


@Equation(v='x4', pV='x2')
def f4(x2, u4):
    return L(x2, u4)


@Equation(v='x5', pV='x3')
def f5(x3, u5):
    return L(x3, u5)


@Equation(v='x6', pV='x4')
def f6(x4, u6):
    return L(x4, u6)


@Equation(v='x7', pV='x5')
def f7(x5, u7):
    return L(x5, u7)


@Equation(v='x8', pV='x6')
def f8(x6, u8):
    return 0.3 * u8 + (s(x6 + 1) - 1)


@Equation(v='x9', pV=['x7', 'x8'])
def f9(x7, x8, u9):
    return cdf_1(-s((x7 * 1.3 + x8) / 3 + 1) + 2, 0.6, u9)


def largebd_nlin_init(*args, **kwarg) -> TensorSCM:
    return TensorSCM(
        name='largebd_nlin',
        equations=[
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            f7,
            f8,
            f9,
        ],
        exogenous_distrs={
            'u1': Uniform(low=0, high=1),
            'u2': Uniform(low=0, high=1),
            'u3': Uniform(low=0, high=1),
            'u4': Uniform(low=0, high=1),
            'u5': Uniform(low=0, high=1),
            'u6': Uniform(low=0, high=1),
            'u7': Uniform(low=0, high=1),
            'u8': Uniform(low=0, high=1),
            'u9': Uniform(low=0, high=1),
        },
    )
