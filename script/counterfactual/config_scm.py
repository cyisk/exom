from dataclasses import dataclass, field
from typing import *
from typing import Any

from common.scm import *
from dataset.synthetic import datasets
from script.config import Config
from script.proxy_scm.causal_nf.config_model import *
from script.proxy_scm.gan_ncm.config_model import *


@dataclass
class SyntheticSCMConfig(Config):
    name: str
    kwargs: dict = field(default_factory=dict)

    def get_scm(self) -> TensorSCM:
        kwargs = self.kwargs or {}
        return datasets[self.name](**kwargs)


@dataclass
class ProxySCMConfig(Config):
    proxy_type: str
    # Base SCM (without access to true mechanism),
    # providing causal graph and endogenous shapes
    base_scm_kwargs: SyntheticSCMConfig
    # Causal normalizing flow
    model_kwargs: CausalNFConfig | GANNCMConfig
    # Model loading
    checkpoint_path: str = None,
    map_location: str = None,

    def serialize(self) -> Dict[str, Any]:
        return {
            'proxy_type': self.proxy_type,
            'base_scm_kwargs': self.base_scm_kwargs.serialize(),
            'model_kwargs': self.model_kwargs.serialize(),
            'checkpoint_path': self.checkpoint_path,
            'map_location': self.map_location,
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        proxy_type = str(config_dict['proxy_type'])
        if proxy_type == 'causal_nf':  # Causal Normalizing Flow
            model_kwargs = CausalNFConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        elif proxy_type == 'gan_ncm':  # GAN NCM
            model_kwargs = GANNCMConfig.deserialize(
                config_dict=config_dict['model_kwargs']
            )
        else:
            raise ValueError("Unsupported Proxy SCM type.")
        return ProxySCMConfig(
            proxy_type=proxy_type,
            base_scm_kwargs=SyntheticSCMConfig.deserialize(
                config_dict['base_scm_kwargs']
            ),
            model_kwargs=model_kwargs,
            checkpoint_path=config_dict['checkpoint_path']
            if 'checkpoint_path' in config_dict else None,
            map_location=config_dict['map_location']
            if 'map_location' in config_dict else None,
        )

    def get_scm(self) -> TensorSCM:
        base_scm = self.base_scm_kwargs.get_scm()
        if self.proxy_type == 'causal_nf':  # Causal Normalizing Flow
            model = self.model_kwargs.load_model(
                base_scm.causal_graph, base_scm.endogenous_dimensions,
                checkpoint_path=self.checkpoint_path,
                map_location=self.map_location,
            )
            return CausalNormalizingFlowSCM(model)
        if self.proxy_type == 'gan_ncm':  # GAN NCM
            U = base_scm.causal_graph.augment().exogenous_nodes
            V = base_scm.causal_graph.augment().endogenous_nodes
            exo_dims = {u: th.Size([sum([base_scm.endogenous_features[v]
                                         for v in base_scm.causal_graph.augment().exo_graph[u]])]) for u in U}
            endo_dims = {v: base_scm.endogenous_dimensions[v] for v in V}
            endo_logits = {v: 2 for v in V}  # binary
            model = self.model_kwargs.load_model(
                base_scm.causal_graph,
                endogenous_dimensions=endo_dims,
                endogenous_logits=endo_logits,
                exogenous_dimensions=exo_dims,
                checkpoint_path=self.checkpoint_path,
                map_location=self.map_location,
            )
            return GANNCMSCM(model)
        else:
            raise ValueError("Unsupported Proxy SCM type.")


@dataclass
class SCMConfig(Config):
    scm_type: str
    scm_kwargs: SyntheticSCMConfig | ProxySCMConfig

    def serialize(self) -> Dict[str, Any]:
        return {
            'scm_type': self.scm_type,
            'scm_kwargs': self.scm_kwargs.serialize(),
        }

    @classmethod
    def deserialize(cls, config_dict: Dict[str, Any]) -> Config:
        scm_type = str(config_dict['scm_type'])
        if scm_type == 'synthetic':  # Synthetic SCM
            scm_kwargs = SyntheticSCMConfig.deserialize(
                config_dict=config_dict['scm_kwargs']
            )
        elif scm_type == 'proxy':  # Proxy SCM
            scm_kwargs = ProxySCMConfig.deserialize(
                config_dict=config_dict['scm_kwargs']
            )
        else:
            raise ValueError("Unsupported SCM type.")
        return SCMConfig(
            scm_type=scm_type,
            scm_kwargs=scm_kwargs,
        )

    def get_scm(self) -> TensorSCM:
        return self.scm_kwargs.get_scm()
