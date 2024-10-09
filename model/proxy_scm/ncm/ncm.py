import torch as th
import torch.nn as nn
import torch.nn.functional as F
from math import prod
from torch.distributions import Independent, Normal, Uniform
from typing import *

from zuko.nn import MLP

from common.graph.causal import *
from common.graph.causal import *
from common.scm.utils import *
from dataset.utils import *


TensorDict = Dict[str, th.Tensor]


def init_exogenous_distribution(
    exogenous_dimensions: Dict[str, th.Size],
    exogenous_distribution_type: str = 'gaussian',
):
    if exogenous_distribution_type == 'gaussian':
        exogenous_distrs = {
            u: Independent(Normal(
                loc=th.zeros(exogenous_dimensions[u]),
                scale=th.ones(exogenous_dimensions[u]),
            ), len(exogenous_dimensions[u]))
            for u in exogenous_dimensions
        }
    elif exogenous_distribution_type == 'uniform':
        exogenous_distrs = {
            u: Independent(Uniform(
                lower=th.zeros(exogenous_dimensions[u]),
                upper=th.ones(exogenous_dimensions[u]),
            ), len(exogenous_dimensions[u]))
            for u in exogenous_dimensions
        }
    else:
        raise NotImplementedError('Unsupported distribution type.')
    return exogenous_distrs


class NCM(nn.Module):
    def __init__(self,
                 causal_graph: DirectedMixedGraph,
                 endogenous_dimensions: Dict[str, th.Size],
                 endogenous_logits: Dict[str, int],  # binary only
                 exogenous_dimensions: Dict[str, th.Size],
                 exogenous_distribution_type: str = 'gaussian',
                 hidden_features: Sequence[int] |
                 Dict[str, Sequence[int]] = [64, 64],
                 ) -> None:
        nn.Module.__init__(self)

        self._causal_graph = causal_graph
        # Note that NCM only works on discrete endogenous variable
        self._endo_dimensions = {
            x: endogenous_dimensions[x]
            for x in sorted(endogenous_dimensions)
        }
        self._exo_dimensions = {
            x: exogenous_dimensions[x]
            for x in sorted(exogenous_dimensions)
        }
        self._exo_distribution_type = exogenous_distribution_type
        self._hidden_features = hidden_features
        self.register_buffer('dummy_val', th.tensor(0))

        # Get parents
        graph = causal_graph.augment()
        pV = inv(graph.endogenous_subgraph)
        self.pV = {v: sorted(pV[v]) for v in pV}
        pU = inv(graph.exogenous_subgraph)
        self.pU = {u: sorted(pU[u]) for u in pU}
        cV = graph.endogenous_subgraph
        U = sorted(graph.exogenous_nodes)

        # The topological order of endogenous
        sccs = tarjan(cV)
        contract_sccs = contract_scc(sccs, cV)
        topological_order = topological_sort(contract_sccs)
        self._sccs = [sccs[i] for i in topological_order]

        # Exogenous distributions
        assert set(U) == set(exogenous_dimensions.keys()), \
            "It is recommanded to augument a causal graph then assign dimension for each node marked as LATENT."
        self._exo_distrs = init_exogenous_distribution(
            exogenous_dimensions=exogenous_dimensions,
            exogenous_distribution_type=exogenous_distribution_type,
        )

        # Dimensions to features
        self._endo_features = {
            v: prod(dim_v) for v, dim_v in self._endo_dimensions.items()
        }
        self._exo_features = {
            u: prod(dim_u) for u, dim_u in self._exo_dimensions.items()
        }

        # Neural proxies
        def proxy_mechanism(v: str):
            feature_1 = sum(self._endo_features[p] for p in pV[v])
            feature_2 = sum(self._exo_features[p] for p in pU[v])
            return MLP(
                in_features=feature_1 + feature_2,
                out_features=self._endo_features[v],
                hidden_features=hidden_features
                if not isinstance(hidden_features, dict)
                else hidden_features[v],
            )
        self._nns = nn.ModuleDict({
            v: proxy_mechanism(v) for v in self._endo_dimensions
        })

    def call_logits(self,
                    U: TensorDict,
                    T: Optional[TensorDict] = None,
                    W_T: Optional[TensorDict] = None,
                    W_E: Optional[TensorDict] = None,
                    ) -> TensorDict:
        batch_shape = batchshape(U, self._exo_dimensions)
        dims = {**self._endo_dimensions, **self._exo_dimensions}

        # Device
        device = self.dummy_val.device
        if not next(iter(U.values())).device == self.dummy_val.device:
            self._exo_distrs = send_distributions_to(
                self._exo_distrs, device=device
            )
            U = {u: u_val.to(device=device) for u, u_val in U.items()}

        def _pad(v, pad_val: float = 0.0):
            return th.full((*batch_shape, *self._endo_dimensions[v]),
                           fill_value=pad_val,
                           device=device)

        # Initialize V
        V = {}
        V = {v: V[v] if v in V else _pad(v).reshape(*batch_shape, -1)
             for v in self._endo_dimensions}

        # Initialize T
        T = {} if T is None else T
        T = {v: T[v] if v in T else _pad(v)
             for v in self._endo_dimensions}

        # Initialize W_T
        W_T = {} if W_T is None else W_T
        W_T = {v: W_T[v] if v in W_T else _pad(v).bool()
               for v in self._endo_dimensions}

        # Initialize W_E
        W_E = {} if W_E is None else W_E
        W_E = {v: W_E[v] if v in W_E else _pad(v, 1).bool()
               for v in self._endo_dimensions}

        # Inference
        for scc in self._sccs:
            for v in scc:
                # Proxy function
                f = self._nns[v]

                # Endogenous and exogenous input
                pV = {x: (V[x] > 0.5) for x in self.pV[v]}  # binary
                pU = {x: U[x] for x in self.pU[v]}
                uv = {**pV, **pU}
                dim_uv = {uv: dim_uv for uv, dim_uv in dims.items()
                          if uv in pV or uv in pU}
                f_input = batch(uv, dim_uv)

                # Inference
                v_val = f(f_input)
                V_val = unbatch(v_val, {v: dims[v]})
                if dims[v] == th.Size():
                    V[v] = V[v].squeeze(-1)
                V[v] = V[v] - V[v].detach() + V_val[v]

                # Intervention
                V[v][W_T[v]] = V[v][W_T[v]] - V[v][W_T[v]].detach() + \
                    T[v][W_T[v]]

        E = {v: V[v] if v in V else _pad(v)
             for v in self._endo_dimensions}
        # E = V \otimes W_E
        for v in self._endo_dimensions:
            E[v][~W_E[v]] = 0
        return E

    def call(self,
             U: TensorDict,
             T: Optional[TensorDict] = None,
             W_T: Optional[TensorDict] = None,
             W_E: Optional[TensorDict] = None,
             soft: bool = False,
             ) -> TensorDict:
        # Note that this call is non-differentiable
        # T = self.expand_for_logits(T) if T is not None else None
        # W_T = self.expand_mask_for_logits(W_T) if W_T is not None else None
        # W_E = self.expand_mask_for_logits(W_E) if W_E is not None else None
        E = self.call_logits(U, T=T, W_T=W_T, W_E=W_E)
        return {
            x: (x_val > 0.5).float()
            if not soft else x_val
            for x, x_val in E.items()
        }

    def batched_call(self,
                     u: th.Tensor,
                     t: Optional[th.Tensor] = None,
                     w_t: Optional[th.Tensor] = None,
                     w_e: Optional[th.Tensor] = None,
                     soft: bool = False,
                     ) -> th.Tensor:
        # Unbatch input tensors, call, then batch result
        return batch(
            self.call(
                U=unbatch(u, self._exo_dimensions),
                T=unbatch(t, self._endo_dimensions)
                if t is not None else None,
                W_T=unbatch(w_t, self._endo_dimensions)
                if w_t is not None else None,
                W_E=unbatch(w_e, self._endo_dimensions)
                if w_e is not None else None,
                soft=soft,
            ),
            self._endo_dimensions if not soft else self._endo_dimensions,
        )

    def forward(self, u: th.Tensor):
        return self.batched_call(u, soft=True)

    def to(self, device: Optional[str | th.device | int] = None, *args, **kwargs):
        self._exo_distrs = send_distributions_to(
            self._exo_distrs, device=device
        )
        return super().to(device=device, *args, **kwargs)

    def noise(self, sample_shape: th.Size | int | Iterable = None) -> TensorDict:
        return {u: ud.sample(shapeit(sample_shape)).to(device=self.dummy_val.device)
                for u, ud in self._exo_distrs.items()}

    def batched_noise(self, sample_shape: th.Size | int | Iterable = None) -> TensorDict:
        return batch(self.noise(sample_shape), self._exo_dimensions)

    """def expand_for_logits(self, X: TensorDict) -> th.Tensor:
        return {
            x: F.one_hot(
                x_val.long(),
                num_classes=self._endo_logits[x],
            ).float()
            for x, x_val in X.items()
        }

    def expand_mask_for_logits(self, X: TensorDict) -> th.Tensor:
        return {
            x: feature_expand(x_val, th.Size((self._endo_logits[x], )))
            for x, x_val in X.items()
        }"""
