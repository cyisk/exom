import copy
import torch as th
from torch.distributions import Distribution
from math import prod
from typing import *

from common.scm.eq import EquationWrapper
from common.scm.utils import *
from common.graph.causal import *


class SCM:
    def __init__(self,
                 equations: List[EquationWrapper],
                 exogenous_distrs: Optional[Dict[str, Distribution]] = None,
                 name: str = 'scm',
                 ) -> None:
        # Name
        self._scm_name = name

        # Set equations
        self._equations = equations
        self._F = {eq.endogenous_output: eq
                   for eq in self._equations}

        # Set exogenous distributions
        self._is_nondeterministic = exogenous_distrs is not None
        if self._is_nondeterministic:
            self._PU = exogenous_distrs

        # Get parent relations
        self._pV = {eq.endogenous_output: eq.endogenous_inputs
                    for eq in self._equations}
        self._pU = {eq.endogenous_output: eq.exogenous_inputs
                    for eq in self._equations}

        def _map_reverse(X: Dict[str, List[str]]) -> Dict[str, List[str]]:
            Y = {}
            for k, vs in X.items():
                for v in vs:
                    if v not in Y:
                        Y[v] = []
                    Y[v].append(k)
            return Y

        # Reverse parent relations to get child relations
        self._cV = {**{v: [] for v in self._pV.keys()},
                    **_map_reverse(self._pV)}
        self._cU = _map_reverse(self._pU)

        # Get endogenous and exogenous variable from equations
        self._V = sorted(list(self._cV.keys()))
        self._U = sorted(list(self._cU.keys()))

        # Checking duplicates in output endogenous variables
        V = set()
        duplicates = [(eq.endogenous_output, eq) for eq in self._equations
                      if eq.endogenous_output in V or V.add(eq.endogenous_output)]
        assert len(duplicates) == 0, \
            f"There are duplicates in equations: {duplicates}"

        # Checking non-exists in input endogenous variables
        nonexists = set(self._V) - set(V)
        assert len(nonexists) == 0, \
            f"There are non-existed endogenous variable in equations: {list(nonexists)}"

        # Checking ambiguoities between endogenous and exogenous variables
        ambiguities = set(self._V).intersection(set(self._U))
        assert len(ambiguities) == 0, \
            f"There are ambiguities in endogenous and exogenous variable names: {list(ambiguities)}."

        # Checking exogenous distributions
        if self._is_nondeterministic:
            nondistributed = set(self._U) - set(self._PU.keys())
            assert len(nondistributed) == 0, \
                f"SCM is nondeterministic, but the following exogenous variables have no distributions: {list(nondistributed)}"

        # The topological order of endogenous (including circles)
        self._sccs = tarjan(self._cV)
        self._contract_sccs = contract_scc(self._sccs, self._cV)
        self._topological_order = topological_sort(self._contract_sccs)
        self._sccs = [self._sccs[i] for i in self._topological_order]

        # Causal graphs
        self._aug_graph_dict = {**self._cV, **self._cU}
        self._aug_graph = AugmentedGraph()
        for u in self._U:
            for v in self._cU[u]:
                self._aug_graph.add_exogenous_edge(u, v)
        for v in self._V:
            for x in self._cV[v]:
                self._aug_graph.add_endogenous_edge(v, x)
        self._causal_graph = self._aug_graph.unaugment()

    def __call__(self,
                 U: Dict[str, Any],
                 T: Dict[str, Any] = None,
                 V_init: Optional[Dict[str, Any]] = None,
                 n_steps: int = 1,
                 ) -> Dict[str, Any]:
        V = {} if V_init is None else copy.deepcopy(V_init)
        T = {} if T is None else T

        for scc in self._sccs:
            for v in scc:
                if v in T:  # Intervention
                    V[v] = T[v]
                else:  # Equation
                    f = self._F[v]
                    f(V, U)

        if n_steps > 1:
            return self.__call__(U, T=T, n_steps=n_steps - 1, V_init=V)

        return V

    def sample(self, *args, **kwargs) -> Dict[str, Any]:
        assert self._is_nondeterministic, "Nondeterministic SCM is required."

        return {u: ud.sample(*args, **kwargs) for u, ud in self._PU.items()}

    def log_prob(self, U: Dict[str, Any]) -> float:
        assert self._is_nondeterministic, "Nondeterministic SCM is required."

        unknowns = set(U.keys()) - set(self._U)
        assert unknowns, \
            f"There are unknown exogenous variables in given U: {list(unknowns)}"

        return sum(ud.log_prob(U[u]) for u, ud in self._PU.items())

    def __str__(self) -> str:
        return '\n'.join([self._scm_name + ':'] + [
            eq.__str__() for eq in self._equations
        ])

    @property
    def name(self) -> str:
        return self._scm_name

    @property
    def equations(self) -> List[EquationWrapper]:
        return self._equations

    @property
    def exogenous_distributions(self) -> Dict[str, Distribution]:
        return self._PU

    @property
    def exogenous_variables(self) -> List[str]:
        return self._U

    @property
    def endogenous_variables(self) -> List[str]:
        return self._V

    @property
    def causal_graph(self) -> DirectedMixedGraph:
        return self._causal_graph

    @property
    def augmented_causal_graph(self) -> AugmentedGraph:
        return self._aug_graph


TensorDict = Dict[str, th.Tensor]


class TensorSCM(SCM):
    def __init__(self,
                 equations: List[EquationWrapper],
                 exogenous_distrs: Optional[Dict[str, Distribution]] = None,
                 name: str = 'scm',
                 device: Optional[str | th.device | int] = None,
                 V_init: Optional[TensorDict] = None,
                 n_steps: int = 1,
                 ) -> None:
        super().__init__(
            equations=equations,
            exogenous_distrs=send_distributions_to(exogenous_distrs, device),
            name=name,
        )
        self._device = device

        # Get dimensions
        U: TensorDict = self.sample()
        self._dim_U = {u: U[u].shape for u in self._U}
        V: TensorDict = super().__call__(
            U, V_init=V_init, n_steps=n_steps
        )
        self._V_init_for_init = V_init
        self._n_steps_for_init = n_steps
        self._dim_V = {v: V[v].shape for v in self._V}

    def to(self, device: Optional[str | th.device | int] = None) -> "TensorSCM":
        if device == self.device:
            return self
        else:
            scm_copy = copy.deepcopy(self)
            scm_copy._PU = send_distributions_to(scm_copy._PU, device)
            return scm_copy

    def __call__(self,
                 U: TensorDict,
                 T: Optional[TensorDict] = None,
                 W_T: Optional[TensorDict] = None,
                 W_E: Optional[TensorDict] = None,
                 V_init: Optional[TensorDict] = None,
                 n_steps: int = 1,
                 ) -> TensorDict:
        batch_shape = batchshape(U, self._dim_U)

        def _pad(v, pad_val: float = 0.0):
            return th.full((*batch_shape, *self._dim_V[v]),
                           fill_value=pad_val,
                           device=self._device)

        # Initialize V
        V = {} if V_init is None else copy.deepcopy(V_init)
        V = {v: V[v] if v in V else _pad(v) for v in self._dim_V}

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

        # Inference
        for scc in self._sccs:
            for v in scc:
                f = self._F[v]
                f(V, U)
                # V = V \otimes (1 - W_T) + T \otimes W_T
                V[v][W_T[v]] = T[v][W_T[v]]

        if n_steps > 1:
            return self.__call__(U, T=T, W_T=W_T, W_E=W_E,
                                 n_steps=n_steps-1, V_init=V)
        else:
            E = {v: V[v] if v in V else _pad(v) for v in self._dim_V}
            # E = V \otimes W_E
            for v in self._dim_V:
                E[v][~W_E[v]] = 0
            return E

    def batched_call(self,
                     u: th.Tensor,
                     t: Optional[th.Tensor] = None,
                     w_t: Optional[th.Tensor] = None,
                     w_e: Optional[th.Tensor] = None,
                     v_init: Optional[th.Tensor] = None,
                     n_steps: int = 1,
                     ) -> th.Tensor:
        # Unbatch input tensors, call, then batch result
        return batch(
            self.__call__(
                U=unbatch(u, self.exogenous_dimensions),
                T=unbatch(t, self.endogenous_dimensions)
                if t is not None else None,
                W_T=unbatch(w_t, self.endogenous_dimensions)
                if w_t is not None else None,
                W_E=unbatch(w_e, self.endogenous_dimensions)
                if w_e is not None else None,
                V_init=unbatch(v_init, self.endogenous_dimensions)
                if v_init is not None else None,
                n_steps=n_steps,
            ),
            self.endogenous_dimensions,
        )

    def sample(self, sample_shape: th.Size | int | Iterable = None) -> TensorDict:
        return {u: ud.sample(shapeit(sample_shape))
                for u, ud in self._PU.items()}

    def batched_sample(self, sample_shape: th.Size | int | Iterable = None) -> th.Tensor:
        return batch(self.sample(sample_shape), self.exogenous_dimensions)

    def rsample(self, sample_shape: th.Size | int | Iterable = None) -> TensorDict:
        return {u: ud.rsample(shapeit(sample_shape))
                for u, ud in self._PU.items()}

    def batched_rsample(self, sample_shape: th.Size | int | Iterable = None) -> th.Tensor:
        return batch(self.rsample(sample_shape), self.exogenous_dimensions)

    def log_prob(self, U: TensorDict) -> th.Tensor:
        assert self._is_nondeterministic, "Nondeterministic SCM is required."

        unknowns = set(U.keys()) - set(self._U)
        assert len(unknowns) == 0, \
            f"There are unknown exogenous variables in given U: {list(unknowns)}"

        return sum([ud.log_prob(U[u]) for u, ud in self._PU.items()])

    def batched_log_prob(self, u: th.Tensor) -> th.Tensor:
        return self.log_prob(unbatch(u, self.exogenous_dimensions))

    @property
    def device(self) -> str | th.device | int:
        return self._device

    @property
    def exogenous_dimensions(self) -> Dict[str, th.Size]:
        return self._dim_U

    @property
    def exogenous_features(self) -> Dict[str, int]:
        return {u: prod(self._dim_U[u]) for u in self._U}

    @property
    def endogenous_dimensions(self) -> Dict[str, th.Size]:
        return self._dim_V

    @property
    def endogenous_features(self) -> Dict[str, int]:
        return {v: prod(self._dim_V[v]) for v in self._V}
