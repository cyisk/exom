from typing import *
import argparse
import os
import re
import shutil
import torch as th

from lightning import seed_everything
from torch.utils.data import DataLoader
from dataset.evidence import BatchedEvidenceDataset, evidence_collate_fn

# Preset candidates
num_joints = [1, 3, 5]
density_estimators = {
    'gmm': 'gmm',
    'maf': 'maf',
    'nsf': 'nsf',
    'ncsf': 'ncsf',
    'nice': 'nice',
    'naf': 'naf',
    'unaf': 'unaf',
    'sospf': 'sospf',
    'bpf': 'bpf',
}
hidden_features = {
    '64': [64],
    '128': [128],
    '64x2': [64, 64],
    '96x2': [96, 96],
    '128x2': [128, 128],
    '192x2': [192, 192],
    '256x2': [256, 256],
    '32x3': [32, 32, 32],
    '64x3': [64, 64, 64],
    '96x3': [96, 96, 96],
}
reduces = ['concat', 'attn', 'wsum', 'sum']

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True)
parser.add_argument('-s', '--scm', type=str, required=True)
parser.add_argument('-j', '--num_joint', type=int)
parser.add_argument('-w', '--workers', type=int)
parser.add_argument('-nv', '--no_validate',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('-sv', '--val_sample_size', type=int)
parser.add_argument('-ev', '--eval_sample_size', type=int)
parser.add_argument('-bv', '--val_batch_size', type=int)
parser.add_argument('-sd', '--seed', type=int)
args = parser.parse_args()

scm = args.scm
experiment_name = args.name or 'default_experiment'
num_joint = args.num_joint or 1
assert num_joint in num_joints
workers = args.workers or 23
no_validate = args.no_validate or False
val_sample_size = args.val_sample_size or 1024
eval_sample_size = args.eval_sample_size or 1024
val_batch_size = args.val_batch_size or 1
seed = args.seed or 0

tmpl = \
    """
scm:
  scm_type: synthetic
  scm_kwargs:
    name: {scm}

evidence:
  evidence_type: context_concat
  evidence_kwargs:
    context_mode: 
    - e
  batched: true
  max_len_joint: {num_joint}

train_dataset:
  sampler:
    sampler_type: mcar_bernoulli
    sampler_kwargs:
      joint_number_low: {num_joint}
      joint_number_high: {num_joint}
      prob_intervened: 0.2
      prob_observed: 0.75
      prob_feature_observed: 1
  size: 16384

train_dataloader:
  batch_size: 256
  num_workers: {workers}

val_dataset:
  sampler:
    sampler_type: mcar_bernoulli
    sampler_kwargs:
      joint_number_low: {num_joint}
      joint_number_high: {num_joint}
      prob_intervened: 0.2
      prob_observed: 0.75
      prob_feature_observed: 1
  size: {val_sample_size}

val_dataloader:
  batch_size: {val_batch_size}
  num_workers: {workers}

model:
  model_type: naive
  model_kwargs:
    indicator_type: l1
    eval_sample_size: {eval_sample_size}

trainer:
  name: {name}
  max_epochs: 1
  check_val_every_n_epoch: 1
  progress_bar_enable: true
  checkpoint_enable: true
  logger_enable: true

seed: {seed}
"""


def config_tmpl(
    name: str = 'default_experiment'
):
    kwargs = {
        'scm': scm,
        'num_joint': num_joint,
        'workers': workers,
        'val_sample_size': val_sample_size,
        'val_batch_size': val_batch_size,
        'eval_sample_size': eval_sample_size,
        'name': name,
        'seed': seed,
    }
    return tmpl.format(**kwargs)


def get_train_result(ckpt_path: str):
    reg_str = 'epoch=(\d+)-loss=(\-?\d+\.\d+)-esp=(\d+\.\d+)'
    reg = re.compile(reg_str)
    if not os.path.exists(ckpt_path):
        return False
    if len(os.listdir(ckpt_path)) == 0:
        return False
    ckpt_name = os.listdir(ckpt_path)[0]
    epoch, loss, esp = reg.search(ckpt_name).groups()
    return {
        'epoch': epoch,
        'loss': loss,
        'esp': esp,
    }


def trial():
    trial_name = f'j={num_joint}'
    if not seed == 0:
        trial_name += f',{seed}'
    name = f'{experiment_name}/{scm}/{trial_name}'
    config_path = f'config/exo_match/{name}.yaml'
    log_path = f'output/{name}/logs'
    ckpt_path = f'output/{name}/checkpoints'

    # Skip complete trial
    res = get_train_result(ckpt_path)
    if isinstance(res, dict):
        epoch = res['epoch']
        if epoch == str(0):
            print(name, 'skipped')
            return

    # Clear incomplete output if exists
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)

    # Generate configuration
    config = config_tmpl(name=name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w+', encoding='utf-8') as f:
        f.write(config)

    # Training
    def callback(val_logs):
        if len(val_logs) == 0:
            return
        else:
            th.save(val_logs, os.path.join(log_path, 'val_logs.pt'))

    from script.counterfactual import train_from_config_file
    val_dataset_path = f'script/counterfactual/density_estimation/val_saves/{scm}_{num_joint}.pt'
    train_from_config_file(
        config_path, callback, with_estimate=True, val_dataset_path=val_dataset_path
    )


if __name__ == '__main__':
    trial()

"""
bash script/counterfactual/utils/sample/train_rs.sh\
    -n "test"\
    -s "simpson_nlin"\
    -j "5"\
    -sv 1024 -bv 1
"""
