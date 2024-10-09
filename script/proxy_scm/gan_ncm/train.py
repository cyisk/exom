import torch
import yaml
from lightning import seed_everything
from typing import *

from script.proxy_scm.gan_ncm.config_dataset import *
from script.proxy_scm.gan_ncm.config_model import ModelConfig
from script.config_dataloader import DefaultDataloaderConfig
from script.config_trainer import DefaultTrainerConfig


def train_from_config(scm_config: SCMConfig,
                      train_dataset_config: ObservationalDatasetConfig,
                      train_dataloader_config: DefaultDataloaderConfig,
                      model_config: ModelConfig,
                      trainer_config: DefaultTrainerConfig,
                      val_dataset_config: ObservationalDatasetConfig = None,
                      val_dataloader_config: DefaultDataloaderConfig = None,
                      seed: int = 0,
                      ):
    seed_everything(seed)
    torch.set_float32_matmul_precision('medium')

    scm = scm_config.get_scm()
    train_dataset = train_dataset_config.get_dataset(scm)
    train_dataloader = train_dataloader_config.get_datasetloader(train_dataset)
    if val_dataloader_config is not None:
        val_dataset = val_dataset_config.get_dataset(scm)
        val_dataloader = val_dataloader_config.get_datasetloader(val_dataset)
    else:
        val_dataloader = None
    # Parameters for binary tensors
    U = scm.causal_graph.augment().exogenous_nodes
    V = scm.causal_graph.augment().endogenous_nodes
    exo_dims = {u: torch.Size([sum([scm.endogenous_features[v]
                                    for v in scm._aug_graph_dict[u]])]) for u in U}
    endo_dims = {v: scm.endogenous_dimensions[v] for v in V}
    endo_logits = {v: 2 for v in V}  # binary
    model = model_config.get_model(
        causal_graph=scm.causal_graph,
        exogenous_dimensions=exo_dims,
        endogenous_dimensions=endo_dims,
        endogenous_logits=endo_logits,
    )
    trainer = trainer_config.get_trainer()

    # Start training
    fit_kwargs = {
        'model': model,
        'train_dataloaders': train_dataloader,
        'val_dataloaders': val_dataloader,
    }
    if model_config.checkpoint_path is not None:
        fit_kwargs['ckpt_path'] = model_config.checkpoint_path
    if trainer_config.check_val_every_n_epoch > trainer_config.max_epochs:
        del fit_kwargs['val_dataloaders']
    trainer.fit(**fit_kwargs)

    return model


def train_from_config_string(config_str: str):
    config_dict = yaml.safe_load(config_str)

    # SCM config
    assert 'scm' in config_dict
    scm_config = SCMConfig.deserialize(config_dict['scm'])

    # train dataset config
    assert 'train_dataset' in config_dict
    train_dataset_config = ObservationalDatasetConfig.deserialize(
        config_dict['train_dataset']
    )
    assert 'train_dataloader' in config_dict
    train_dataloader_config = DefaultDataloaderConfig.deserialize(
        config_dict['train_dataloader']
    )

    # validation dataset config
    if 'val_dataset' in config_dict:
        assert 'val_dataloader' in config_dict
        val_dataset_config = ObservationalDatasetConfig.deserialize(
            config_dict['val_dataset']
        )
        val_dataloader_config = DefaultDataloaderConfig.deserialize(
            config_dict['val_dataloader']
        )
    else:
        val_dataset_config = val_dataloader_config = None

    # model config
    assert 'model' in config_dict
    model_config = ModelConfig.deserialize(config_dict['model'])

    # trainer config
    assert 'trainer' in config_dict
    trainer_config = DefaultTrainerConfig.deserialize(config_dict['trainer'])
    trainer_config.checkpoint_name = '{epoch}-{g_loss:.3f}-{d_loss:.3f}'

    # seed
    if 'seed' in config_dict:
        seed = config_dict['seed']
    else:
        seed = 0

    # Call config train
    return train_from_config(
        scm_config=scm_config,
        train_dataset_config=train_dataset_config,
        train_dataloader_config=train_dataloader_config,
        model_config=model_config,
        trainer_config=trainer_config,
        val_dataset_config=val_dataset_config,
        val_dataloader_config=val_dataloader_config,
        seed=seed,
    )


def train_from_config_file(config_path: str):
    with open(config_path, mode='r', encoding='utf-8') as f:
        yaml_str = f.read()
    return train_from_config_string(yaml_str)


"""
scm:
  scm_type: synthetic
  scm_kwargs:
    name: fairness_xw

train_dataset:
  size: 32768

train_dataloader:
  batch_size: 256
  shuffle: true
  num_workers: 23

model:
  model_type: gan_ncm
  model_kwargs:
    ncm_hidden_features:
      - 64
      - 64
    critic_hidden_features:
      - 64
      - 64
    exogenous_distribution_type: gaussian
    learning_rate: 0.001

trainer:
  name: ...
  max_epochs: 200
  progress_bar_enable: true
  checkpoint_enable: true
  logger_enable: true

seed: 0
"""
