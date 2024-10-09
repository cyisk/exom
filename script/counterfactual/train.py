import torch
import yaml
import os
from lightning import seed_everything
from typing import *

from dataset.evidence import *
from script.counterfactual.config_dataset import EvidenceDatasetConfig
from script.counterfactual.config_evidence import EvidenceConfig
from script.counterfactual.config_model import ModelConfig
from script.counterfactual.config_scm import SCMConfig
from script.config_dataloader import DefaultDataloaderConfig
from script.config_trainer import DefaultTrainerConfig


def train_from_config(scm_config: SCMConfig,
                      evidence_config: EvidenceConfig,
                      train_dataset_config: EvidenceDatasetConfig,
                      train_dataloader_config: DefaultDataloaderConfig,
                      model_config: ModelConfig,
                      trainer_config: DefaultTrainerConfig,
                      val_dataset_config: EvidenceDatasetConfig = None,
                      val_dataloader_config: DefaultDataloaderConfig = None,
                      seed: int = 0,
                      callback: Callable = None,
                      on_before_init_callback: Callable = None,
                      on_before_train_callback: Callable = None,
                      val_dataset_path: str = None,
                      with_estimate: bool = False,
                      ):
    seed_everything(seed)
    torch.set_float32_matmul_precision('medium')

    scm = scm_config.get_scm()

    # Before initialize
    if on_before_init_callback is not None:
        on_before_init_callback({
            'scm': scm,
            'scm_confg': scm_config,
            'evidence_config': evidence_config,
            'train_dataset_config': train_dataset_config,
            'train_dataloader_config': train_dataloader_config,
            'model_config': model_config,
            'trainer_config': trainer_config,
            'val_dataset_config': val_dataset_config,
            'val_dataloader_config': val_dataloader_config,
        })

    train_dataset = train_dataset_config.get_dataset(scm, evidence_config)
    train_dataloader = train_dataloader_config.get_datasetloader(
        train_dataset, collate_fn=evidence_collate_fn,
    )
    if val_dataloader_config is not None:
        val_dataset = val_dataset_config.get_dataset(scm, evidence_config)
        if val_dataset_path is not None:
            if os.path.exists(val_dataset_path):
                val_dataset.load(val_dataset_path)
            else:
                os.makedirs(os.path.dirname(val_dataset_path), exist_ok=True)
                val_dataset.save(val_dataset_path)
        val_dataloader = val_dataloader_config.get_datasetloader(
            val_dataset, collate_fn=evidence_collate_fn,
        )
    else:
        val_dataloader = None
    model = model_config.get_model(
        scm, evidence_config, train_dataset.mean, train_dataset.std
    )
    if with_estimate:
        model.set_val_estimate(True)
    trainer = trainer_config.get_trainer([FlushCallback()])

    # Before training
    if on_before_train_callback is not None:
        on_before_train_callback({
            'scm': scm,
            'evidence_type': evidence_config.get_evidence_type(),
            'evidence_kwargs': evidence_config.get_evidence_kwargs(),
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'trainer': trainer,
            'model': model,
        })

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
    val_logs = trainer.val_logs

    # Callback
    if callback is not None:
        callback(val_logs)

    return {
        'scm': scm,
        'evidence_type': evidence_config.get_evidence_type(),
        'evidence_kwargs': evidence_config.get_evidence_kwargs(),
        'train_dataset': train_dataset,
        'train_dataloader': train_dataloader,
        'val_dataset': val_dataset,
        'val_dataloader': val_dataloader,
        'trainer': trainer,
        'model': model,
    }


def train_from_config_string(config_str: str,
                             callback: Callable = None,
                             on_before_init_callback: Callable = None,
                             on_before_train_callback: Callable = None,
                             with_estimate: bool = False,
                             val_dataset_path: str = None,
                             ):
    config_dict = yaml.safe_load(config_str)

    # SCM config
    assert 'scm' in config_dict
    scm_config = SCMConfig.deserialize(config_dict['scm'])

    # evidence config
    assert 'evidence' in config_dict
    evidence_config = EvidenceConfig.deserialize(config_dict['evidence'])

    # train dataset config
    assert 'train_dataset' in config_dict
    train_dataset_config = EvidenceDatasetConfig.deserialize(
        config_dict['train_dataset']
    )
    assert 'train_dataloader' in config_dict
    train_dataloader_config = DefaultDataloaderConfig.deserialize(
        config_dict['train_dataloader']
    )

    # validation dataset config
    if 'val_dataset' in config_dict:
        assert 'val_dataloader' in config_dict
        val_dataset_config = EvidenceDatasetConfig.deserialize(
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
    trainer_config.checkpoint_name = '{epoch}-{loss:.3f}-{esp:.3f}'

    # seed
    if 'seed' in config_dict:
        seed = config_dict['seed']
    else:
        seed = 0

    # Call config train
    return train_from_config(
        scm_config=scm_config,
        evidence_config=evidence_config,
        train_dataset_config=train_dataset_config,
        train_dataloader_config=train_dataloader_config,
        model_config=model_config,
        trainer_config=trainer_config,
        val_dataset_config=val_dataset_config,
        val_dataloader_config=val_dataloader_config,
        callback=callback,
        on_before_init_callback=on_before_init_callback,
        on_before_train_callback=on_before_train_callback,
        with_estimate=with_estimate,
        val_dataset_path=val_dataset_path,
        seed=seed,
    )


def train_from_config_file(config_path: str,
                           callback: Callable = None,
                           on_before_init_callback: Callable = None,
                           on_before_train_callback: Callable = None,
                           with_estimate: bool = False,
                           val_dataset_path: str = None,
                           ):
    with open(config_path, mode='r', encoding='utf-8') as f:
        yaml_str = f.read()
    return train_from_config_string(
        yaml_str,
        callback=callback,
        on_before_init_callback=on_before_init_callback,
        on_before_train_callback=on_before_train_callback,
        with_estimate=with_estimate,
        val_dataset_path=val_dataset_path
    )


"""
scm:
  scm_type: synthetic
  scm_kwargs:
    name: simpson_nlin

evidence:
  evidence_type: context_masked
  evidence_kwargs:
    context_mode: 
    - e+t
    - w_e
    - w_t
    mask_mode:
    - mb
    - mb
    - em
  batched: true
  max_len_joint: 3

train_dataset:
  sampler:
    sampler_type: mcar_bernoulli
    sampler_kwargs:
      joint_number_low: 3
      joint_number_high: 3
      prob_intervened: 0.2
      prob_observed: 0.75
      prob_feature_observed: 1
  size: 16384

train_dataloader:
  batch_size: 256
  num_workers: 23

val_dataset:
  sampler:
    sampler_type: mcar_bernoulli
    sampler_kwargs:
      joint_number_low: 3
      joint_number_high: 3
      prob_intervened: 0.2
      prob_observed: 0.75
      prob_feature_observed: 1
  size: 8192

val_dataloader:
  batch_size: 32
  num_workers: 23

model:
  model_type: exom
  model_kwargs:
    density_estimator_type: ncsf
    density_estimator_kwargs:
      transforms: 5
      hidden_features:
        - 128
        - 128
      reduce: wbsum
    base_distribution_type: gaussian
    indicator_type: l1
    eval_sample_size: 1000
    learning_rate: 0.001


trainer:
  name: ...
  max_epochs: 200
  check_val_every_n_epoch: 200
  progress_bar_enable: true
  checkpoint_enable: false
  logger_enable: false

seed: 0
"""
