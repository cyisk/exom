import torch as th
from torch.distributions import Distribution
from typing import *

from common.scm import *
from common.scm.eq import *
from common.scm.utils import *
from model.proxy_scm.ncm import GANNCM
from model.zuko import *

TensorDict = Dict[str, th.Tensor]


class GANNCMSCM(TensorSCM):
    def __init__(self,
                 gan_ncm: GANNCM,
                 name: str = 'gan_ncm',
                 ) -> None:
        self._gan_ncm = gan_ncm
        self._ncm = gan_ncm._ncm

        # Create dummy equations
        exo_graph = self._ncm._causal_graph.augment().exo_graph
        endo_graph = self._ncm._causal_graph.augment().endo_graph
        exo_inv_graph = {u: [] for u in endo_graph}
        for u in exo_graph:
            for v in exo_graph[u]:
                if v not in exo_inv_graph:
                    exo_inv_graph[v] = []
                exo_inv_graph[v].append(u)
        endo_inv_graph = {u: [] for u in endo_graph}
        for u in endo_graph:
            for v in endo_graph[u]:
                if v not in endo_inv_graph:
                    exo_inv_graph[v] = []
                endo_inv_graph[v].append(u)
        dummy_equations = []
        for v in endo_graph:
            pV_in = ', '.join(endo_inv_graph[v])
            if len(pV_in) > 0:
                pV_in += ', '
            pU_in = ', '.join(exo_inv_graph[v])
            func_str = '\n'.join([
                f"def dummy_func({pV_in}{pU_in}):",
                f"    pass",
            ])
            locals = {}
            exec(func_str, globals(), locals)
            eq = Equation(v=v, pV=endo_inv_graph[v])(locals['dummy_func'])
            dummy_equations.append(eq)

        # Create dummy exogenous distribution
        dummy_exogenous_distrs = {
            u: None for u in gan_ncm._ncm._exo_dimensions.keys()
        }

        # Initialization
        SCM.__init__(
            self,
            equations=dummy_equations,
            exogenous_distrs=dummy_exogenous_distrs,
            name=name,
        )
        self._device = gan_ncm.device  # device
        # Flow dimensions is equal
        self._dim_U = gan_ncm._ncm._exo_dimensions
        self._dim_V = gan_ncm._endo_dimensions

    def to(self, device: Optional[str | th.device | int] = None) -> "GANNCMSCM":
        return GANNCMSCM(
            gan_ncm=self._gan_ncm.to(device=device),
            name=self._scm_name,
        )

    def __call__(self,
                 U: TensorDict,
                 T: Optional[TensorDict] = None,
                 W_T: Optional[TensorDict] = None,
                 W_E: Optional[TensorDict] = None,
                 ) -> TensorDict:
        return self._ncm.call(U, T, W_T, W_E, soft=False)

    def batched_call(self,
                     u: th.Tensor,
                     t: th.Tensor = None,
                     w_t: th.Tensor = None,
                     w_e: th.Tensor = None,
                     ) -> th.Tensor:
        return self._ncm.batched_call(u, t, w_t, w_e, soft=False)

    def sample(self,
               sample_shape: th.Size | int | Iterable = None,
               ) -> TensorDict:
        return self._ncm.noise(sample_shape)

    def batched_sample(self,
                       sample_shape: th.Size | int | Iterable = None,
                       ) -> th.Tensor:
        return self._ncm.batched_noise(sample_shape)

    def rsample(self,
                sample_shape: th.Size | int | Iterable = None,
                ) -> TensorDict:
        return {u: ud.rsample(shapeit(sample_shape)).to(device=self._gan_ncm.device)
                for u, ud in self._ncm._exo_distrs.items()}

    def batched_rsample(self,
                        sample_shape: th.Size | int | Iterable = None,
                        ) -> th.Tensor:
        return batch(self.rsample(sample_shape), self._ncm._exo_dimensions)

    def log_prob(self, U: TensorDict) -> th.Tensor:
        return sum([send_distribution_to(ud, U[u].device).log_prob(U[u]) for u, ud in self._ncm._exo_distrs.items()])

    def batched_log_prob(self, u: th.Tensor) -> th.Tensor:
        return self.log_prob(unbatch(u, self._ncm._exo_dimensions))

    @property
    def exogenous_distributions(self) -> Distribution:
        return self._ncm._exo_distrs
