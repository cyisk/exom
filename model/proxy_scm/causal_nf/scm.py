import torch as th
from torch.distributions import Distribution
from typing import *

from common.scm import *
from common.scm.eq import *
from common.scm.utils import *
from model.proxy_scm.causal_nf import CausalNormalizingFlow
from model.zuko import *

TensorDict = Dict[str, th.Tensor]


class CausalNormalizingFlowSCM(TensorSCM):
    def __init__(self,
                 causal_nf: CausalNormalizingFlow,
                 name: str = 'causal_nf',
                 ) -> None:
        self._causal_nf = causal_nf

        # Create dummy equations
        inv_graph = {u: [] for u in causal_nf._graph}
        for u in causal_nf._graph:
            for v in causal_nf._graph[u]:
                if v not in inv_graph:
                    inv_graph[v] = []
                inv_graph[v].append(u)
        dummy_equations = []
        for v in causal_nf._graph:
            pV_in = ', '.join(inv_graph[v])
            if len(pV_in) > 0:
                pV_in += ', '
            func_str = '\n'.join([
                f"def dummy_func({pV_in}u_{v}):",
                f"    pass",
            ])
            locals = {}
            exec(func_str, globals(), locals)
            eq = Equation(v=v, pV=inv_graph[v])(locals['dummy_func'])
            dummy_equations.append(eq)

        # Create dummy exogenous distribution
        dummy_exogenous_distrs = {
            f"u_{v}": None for v in inv_graph
        }

        # Initialization
        SCM.__init__(
            self,
            equations=dummy_equations,
            exogenous_distrs=dummy_exogenous_distrs,
            name=name,
        )
        self._device = causal_nf.device  # device
        # Flow dimensions is equal
        self._dim_U = {
            f'u_{v}': dim_v
            for v, dim_v in causal_nf._endo_dimensions.items()
        }
        self._dim_V = causal_nf._endo_dimensions

    def to(self, device: Optional[str | th.device | int] = None) -> "CausalNormalizingFlowSCM":
        return CausalNormalizingFlowSCM(
            causal_nf=self._causal_nf.to(device=device),
            name=self._scm_name,
        )

    def causal_nf_inverse(self, u: th.Tensor):
        p: BMAF = self._causal_nf()
        v = self._causal_nf.destandardize(p.transform.inv(u))
        return v

    def causal_nf_forward(self, v: th.Tensor):
        p: BMAF = self._causal_nf()
        u = p.transform(self._causal_nf.standardize(v))
        return u

    def __call__(self,
                 U: TensorDict,
                 T: Optional[TensorDict] = None,
                 W_T: Optional[TensorDict] = None,
                 W_E: Optional[TensorDict] = None,
                 ) -> TensorDict:
        batch_size = batchshape(U, self._dim_U)

        def _pad(v, pad_val: float = 0.0):
            return th.full((*batch_size, *self._dim_V[v]),
                           fill_value=pad_val,
                           device=self._device)

        # Initialize T
        T = {} if T is None else T
        T = {v: T[v] if v in T else _pad(v) for v in self._dim_V}

        # Initialize W_T
        W_T = {} if W_T is None else W_T
        W_T = {v: W_T[v] if v in W_T else _pad(v).bool()
               for v in self._dim_V}

        # Initialize W_E
        W_E = {} if W_E is None else W_E
        W_E = {v: W_E[v] if v in W_E else _pad(v, 1).bool()
               for v in self._dim_V}

        # Batched call
        u = batch(U, self.exogenous_dimensions)
        t = batch(T, self.endogenous_dimensions)
        w_t = batch(W_T, self.endogenous_dimensions)
        w_e = batch(W_E, self.endogenous_dimensions)
        e = self.batched_call(u, t, w_t, w_e)

        return unbatch(e, self.endogenous_dimensions)

    def batched_call(self,
                     u: th.Tensor,
                     t: th.Tensor = None,
                     w_t: th.Tensor = None,
                     w_e: th.Tensor = None,
                     ) -> th.Tensor:
        v = self.causal_nf_inverse(u)
        if t is None:
            t = th.zeros_like(v)
        if w_t is None:
            w_t = th.zeros_like(v).bool()
        if w_e is None:
            w_e = th.ones_like(v).bool()
        v[w_t] = t[w_t]
        u_t = self.causal_nf_forward(v)
        u_t[~w_t] = u[~w_t]
        v_t = self.causal_nf_inverse(u_t)
        v_t[~w_e] = 0
        return v_t

    def sample(self,
               sample_shape: th.Size | int | Iterable = None,
               ) -> TensorDict:
        return unbatch(
            self.batched_sample(sample_shape),
            self.exogenous_dimensions,
        )

    def batched_sample(self,
                       sample_shape: th.Size | int | Iterable = None,
                       ) -> th.Tensor:
        p: BMAF = self._causal_nf()
        base: Distribution = p.base
        return base.sample(shapeit(sample_shape))

    def rsample(self,
                sample_shape: th.Size | int | Iterable = None,
                ) -> TensorDict:
        return unbatch(
            self.batched_sample(sample_shape),
            self.exogenous_dimensions,
        )

    def batched_rsample(self,
                        sample_shape: th.Size | int | Iterable = None,
                        ) -> th.Tensor:
        p: BMAF = self._causal_nf()
        base: Distribution = p.base
        return base.rsample(shapeit(sample_shape))

    def log_prob(self, U: TensorDict) -> th.Tensor:
        return self.batched_log_prob(
            batch(U, self.exogenous_dimensions),
        )

    def batched_log_prob(self, u: th.Tensor) -> th.Tensor:
        p: BMAF = self._causal_nf()
        base: Distribution = p.base
        return base.log_prob(u)

    @property
    def exogenous_distributions(self) -> Distribution:
        p: BMAF = self._causal_nf()
        base: Distribution = p.base
        return base
