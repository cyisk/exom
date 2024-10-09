from typing import *
import argparse
import os
import re
import shutil
import torch as th

from script.proxy_scm.causal_nf import train_from_config_file

# Preset candidates
flow_models = {
    'maf': 'maf',
    'nsf': 'nsf',
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

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, required=True)
parser.add_argument('-s', '--scm', type=str, required=True)
parser.add_argument('-m', '--flow_model', type=str)
parser.add_argument('-d', '--hidden_features', type=str)
parser.add_argument('-e', '--max_epochs', type=int)
parser.add_argument('-l', '--checkpoint_path', type=str)
parser.add_argument('-w', '--workers', type=int)
parser.add_argument('-nv', '--no_validate',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('-bv', '--val_batch_size', type=int)
parser.add_argument('-ev', '--val_every_n_epochs', type=int)
parser.add_argument('-sd', '--seed', type=int)
args = parser.parse_args()

scm = args.scm
experiment_name = args.name or 'default_experiment'
flow_model = args.flow_model or 'maf'
assert flow_model in flow_models
hidden_feature = args.hidden_features or '128x2'
max_epoch = args.max_epochs or 200
checkpoint_path = args.checkpoint_path
workers = args.workers or 23
no_validate = args.no_validate or False
val_batch_size = args.val_batch_size or 32
val_every_n_epochs = args.val_every_n_epochs or max_epoch
seed = args.seed or 0

tmpl = \
    """
scm:
  scm_type: synthetic
  scm_kwargs:
    name: {scm}

train_dataset:
  size: 32768

train_dataloader:
  batch_size: 4096
  shuffle: true
  num_workers: {workers}

val_dataset:
  size: 16384

val_dataloader:
  batch_size: {val_batch_size}
  num_workers: {workers}

model:
  model_type: causal_nf
  model_kwargs:
    density_estimator_type: {flow_model}
    density_estimator_kwargs:
      hidden_features:
{hidden_features}
    base_distribution_type: gaussian
    learning_rate: 0.001
{checkpoint_path}

trainer:
  name: {name}
  max_epochs: {max_epoch}
  check_val_every_n_epoch: {val_check_epoch}
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
        'flow_model': flow_model,
        'hidden_features': '\n'.join([' '*8 + '- ' + str(hdim) for hdim in hidden_features[hidden_feature]]),
        'checkpoint_path': f'  checkpoint_path: {checkpoint_path}' if checkpoint_path is not None else '',
        'workers': workers,
        'max_epoch': max_epoch,
        'val_check_epoch': max_epoch + 1 if no_validate else val_every_n_epochs,
        'val_batch_size': val_batch_size,
        'name': name,
        'seed': seed,
    }
    return tmpl.format(**kwargs)


def get_train_result(ckpt_path: str):
    reg_str = 'epoch=(\d+)-loss=(\-?\d+\.\d+)-mmd=(\d+\.\d+)'
    reg = re.compile(reg_str)
    if not os.path.exists(ckpt_path):
        return False
    if len(os.listdir(ckpt_path)) == 0:
        return False
    ckpt_name = os.listdir(ckpt_path)[0]
    epoch, loss, mmd = reg.search(ckpt_name).groups()
    return {
        'epoch': epoch,
        'loss': loss,
        'mmd': mmd,
    }


def trial():
    trial_name = f'm={flow_model},h={hidden_feature}'
    if not seed == 0:
        trial_name += f',{seed}'
    name = f'{experiment_name}/{scm}/{trial_name}'
    config_path = f'config/causal_nf/{name}.yaml'
    log_path = f'output/{name}/logs'
    ckpt_path = f'output/{name}/checkpoints'

    # Skip complete trial
    res = get_train_result(ckpt_path)
    if isinstance(res, dict):
        epoch = res['epoch']
        if epoch == str(max_epoch-1):
            print(trial_name, 'skipped')
            return

    # Clear incomplete output if exists
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)

    # Generate configuration
    config = config_tmpl(name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w+', encoding='utf-8') as f:
        f.write(config)

    # Training
    train_from_config_file(config_path)


if __name__ == '__main__':
    trial()

"""
bash script/proxy_scm/causal_nf/utils/train_causal_nf.sh\
    -n "test"\
    -s "simpson_nlin"\
    -m nsf -d 32x3\
    -bv 4096 -ev 10\
    -e 100
"""
