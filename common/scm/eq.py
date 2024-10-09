import copy
import inspect
from typing import List, Dict, Any, Optional, Callable


class EquationWrapper:

    def __init__(self,
                 f: Callable,
                 v: str,
                 ud_pV_map: Dict[str, str],
                 ud_pU_map: Dict[str, str],
                 ) -> None:
        self._p: EquationWrapper = None
        self._f = f
        self._v = v
        self._pV_map = {}
        self._pU_map = {}

        # Checking ambiguoities in user-defined mappings
        ambiguities = set(ud_pV_map.keys()).intersection(set(ud_pU_map.keys()))
        assert len(ambiguities) == 0, \
            f"There are ambiguities in parameter names mapped to endogenous and exogenous variable names: {list(ambiguities)}."

        params = set(inspect.signature(self._f).parameters.keys())
        free_params = params.copy()
        # Mapping user-defined endogenous symbols
        for pv_param, pv in ud_pV_map.items():
            assert pv_param in params, f"Parameters '{pv_param}' does not exist in function {self._f.__name__}({', '.join(params)})."
            free_params.remove(pv_param)
            self._pV_map[pv_param] = pv
        # Mapping user-defined exogenous symbols
        for pu_param, pu in ud_pU_map.items():
            assert pu_param in params, f"Parameters '{pu_param}' does not exist in function {self._f.__name__}({', '.join(params)})."
            free_params.remove(pu_param)
            self._pU_map[pu_param] = pu
        # Mapping free parameters as exogenous symbols
        for param in free_params:
            self._pU_map[param] = param

        assert len(self._pU_map) > 0, \
            "At least one exogenous variable is required."

    def __str__(self) -> str:
        fname = self._f.__name__ if hasattr(self._f, '__name__')\
            else self._f.__class__.__name__
        return f"{self._v} := {fname}("\
            + f"{', '.join(self._pV_map.values())}"\
            + f"{', ' if len(self._pV_map) > 0 else ''}"\
            + f"{', '.join(self._pU_map.values())})"

    def __call__(self,
                 V: Dict[str, Any],
                 U: Dict[str, Any],
                 ) -> None:
        def _to_kwargs(g_map: Dict[str, str], p_map: Dict[str, str], assert_str: Callable):
            kwargs = {}
            for param, var in p_map.items():
                # Check if variable does exist in context
                assert var in g_map, assert_str(var)
                # In case the function change properties of inputs
                kwargs[param] = copy.deepcopy(g_map[var])
            return kwargs

        kwargs_v = _to_kwargs(V, self._pV_map,
                              lambda var: f"Variable '{var}' does not exist in current endogenous variables: {V.keys()}.")
        kwargs_u = _to_kwargs(U, self._pU_map,
                              lambda var: f"Variable '{var}' does not exist in current exogenous variables: {U.keys()}.")
        kwargs = {**kwargs_v, **kwargs_u}

        v_val = self._f(**kwargs)
        V[self._v] = v_val

    @property
    def endogenous_output(self) -> str:
        if self._p is not None:
            return self._p.endogenous_output
        return self._v

    @property
    def endogenous_inputs(self) -> List[str]:
        if self._p is not None:
            return self._p.endogenous_inputs
        return list(set(self._pV_map.values()))

    @property
    def endogenous_parameter_mapping(self) -> Dict[str, str]:
        if self._p is not None:
            return self._p.endogenous_parameter_mapping
        return self._pV_map

    @property
    def exogenous_inputs(self) -> List[str]:
        if self._p is not None:
            return self._p.exogenous_inputs
        return list(set(self._pU_map.values()))

    @property
    def exogenous_parameter_mapping(self) -> Dict[str, str]:
        if self._p is not None:
            return self._p.exogenous_parameter_mapping
        return self._pU_map


class Equation:

    def __init__(self,
                 v: str,
                 pV: Optional[str | List[str] | Dict[str, str]] = None,
                 pU: Optional[str | List[str] | Dict[str, str]] = None,
                 ) -> None:
        self._v = v

        def _to_dict(X: Optional[str | List[str] | Dict[str, str]]):
            if isinstance(X, dict):
                return X
            elif isinstance(X, list):
                return {x: x for x in X}
            elif isinstance(X, str):
                return {X: X}
            elif X is None:
                return {}
            assert "Invalid type."

        self._ud_pV_map = _to_dict(pV)
        self._ud_pU_map = _to_dict(pU)

    def __call__(self, f: Callable) -> Callable:
        decorator = EquationWrapper(f,
                                    self._v, self._ud_pV_map, self._ud_pU_map)
        return decorator
