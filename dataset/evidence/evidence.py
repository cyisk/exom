import abc
import torch as th
from typing import *

from common.scm import *
from dataset.utils import *


TensorDict = Dict[str, th.Tensor]
SetDict = Dict[str, Set[str]]


def pad_variable(variables: TensorDict,
                 dimensions: Dict[str, th.Size],
                 batch_shape: th.Size = th.Size(),
                 ) -> TensorDict:
    assert set(variables.keys()).issubset(dimensions.keys())
    dimensions = {v: dimensions[v] for v in sorted(dimensions)}
    return {
        v: variables[v] if v in variables
        else th.zeros((*batch_shape, *dim))
        for v, dim in dimensions.items()
    }


def pad_variable_mask(variables: Set[str],
                      dimensions: Dict[str, th.Size],
                      batch_shape: th.Size = th.Size(),
                      ) -> TensorDict:
    assert variables.issubset(dimensions.keys())
    dimensions = {v: dimensions[v] for v in sorted(dimensions)}
    return {
        v: th.ones((*batch_shape, *dim)).bool() if v in variables
        else th.zeros((*batch_shape, *dim)).bool()
        for v, dim in dimensions.items()
    }


class Evidence(abc.ABC):
    def __init__(self,
                 scm: TensorSCM,
                 observation: Optional[TensorDict] = None,
                 intervention: Optional[TensorDict] = None,
                 feature_observed: Optional[TensorDict] = None,
                 e: Optional[th.Tensor] = None,
                 t: Optional[th.Tensor] = None,
                 w_e: Optional[th.Tensor] = None,
                 w_t: Optional[th.Tensor] = None,
                 ) -> None:
        if observation is None:
            # Using copy initialization
            assert e is not None and t is not None and w_e is not None and w_t is not None
            self._e = e
            self._t = t
            self._w_e = w_e
            self._w_t = w_t
            return

        observation = observation or {}
        intervention = intervention or {}
        feature_observed = feature_observed or {}

        self._scm = scm
        # intervened is also observed
        observation = {**observation, **intervention}

        self._observation = observation
        self._intervention = intervention
        self._feature_observed = feature_observed

        # Shape check
        batch_shape = batchshape(observation, scm.endogenous_dimensions)
        assert len(observation) == 0 or len(batch_shape) == 0, \
            "Expect no batch"

        # Manipulated variables
        self._observed = set(observation.keys())
        self._intervened = set() if intervention is None else set(intervention.keys())

        # Create observation mask
        W_E = pad_variable_mask(self._observed,
                                self._scm.endogenous_dimensions)
        W_E = {
            v: W_E[v] & feature_observed[v]
            if v in feature_observed
            else W_E[v] for v in W_E
        }
        self._w_e = batch(W_E, scm.endogenous_dimensions)

        # Create intervention mask
        W_T = pad_variable_mask(self._intervened,
                                self._scm.endogenous_dimensions)
        self._w_t = batch(W_T, scm.endogenous_dimensions)

        # Observation
        padded_observation = pad_variable(
            observation, scm.endogenous_dimensions
        )
        e = batch(padded_observation, scm.endogenous_dimensions)
        self._e = e
        self._e[~self._w_e] = 0

        # Intervention
        padded_intervention = pad_variable(
            intervention, scm.endogenous_dimensions
        )
        t = batch(padded_intervention, scm.endogenous_dimensions)
        self._t = th.zeros_like(e) if t is None else t
        self._t[~self._w_t] = 0

    @property
    def scm(self) -> TensorSCM:
        return self._scm

    @property
    def observed(self) -> Set[str]:
        if not hasattr(self, '_observed'):
            self._observed = indicator_to_set(
                self.w_e, self.scm.endogenous_features
            )
        return self._observed

    @property
    def observation(self) -> TensorDict:
        if not hasattr(self, '_observation'):
            observed = self.observed
            self._observation = {
                v: v_val for v, v_val in self.E
                if v in observed
            }
        return self._observation

    @property
    def E(self) -> TensorDict:
        if not hasattr(self, '_E'):
            self._E = unbatch(self._e, self._scm.endogenous_dimensions)
        return self._E

    @property
    def e(self) -> th.Tensor:
        return self._e

    @property
    def intervened(self) -> Set[str]:
        if not hasattr(self, '_intervened'):
            self._intervened = indicator_to_set(
                self.w_t, self.scm.endogenous_features
            )
        return self._intervened

    @property
    def intervention(self) -> TensorDict:
        if not hasattr(self, '_observation'):
            intervened = self.intervened
            self._intervention = {
                v: v_val for v, v_val in self.E
                if v in intervened
            }
        return self._intervention

    @property
    def T(self) -> TensorDict:
        if not hasattr(self, '_T'):
            self._T = unbatch(self._t, self._scm.endogenous_dimensions)
        return self._T

    @property
    def t(self) -> th.Tensor:
        return self._t

    @property
    def W_E(self) -> TensorDict:
        if not hasattr(self, '_W_E'):
            self._W_E = unbatch(self._w_e, self._scm.endogenous_dimensions)
        return self._W_E

    @property
    def w_e(self) -> th.Tensor:
        return self._w_e

    @property
    def W_T(self) -> TensorDict:
        if not hasattr(self, '_W_T'):
            self._W_T = unbatch(self._w_t, self._scm.endogenous_dimensions)
        return self._W_T

    @property
    def w_t(self) -> th.Tensor:
        return self._w_t

    @property
    def context(self) -> th.Tensor:
        raise NotImplementedError

    @property
    def adjacency(self) -> th.Tensor | None:
        raise NotImplementedError

    @staticmethod
    def context_features(scm: TensorSCM, *args, **kwargs) -> int:
        raise NotImplementedError

    @staticmethod
    def standardize(context: th.Tensor,
                    w_e: th.Tensor,
                    w_t: th.Tensor,
                    prior_mean: th.Tensor,
                    prior_std: th.Tensor,
                    *args,
                    **kwargs) -> th.Tensor:
        raise NotImplementedError
