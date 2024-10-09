from typing import *
import argparse
import os
import re
import shutil
import torch as th

# Preset candidates
num_joints = [1, 3, 5]
density_estimators = {
    'gmm': 'gmm',
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
parser.add_argument('-c', '--context', nargs='+', type=str)
parser.add_argument('-k', '--mask', nargs='+', type=str)
parser.add_argument('-r', '--reduce', type=str)
parser.add_argument('-t', '--structures', type=int)
parser.add_argument('-d', '--hidden_features', type=str)
parser.add_argument('-e', '--max_epochs', type=int)
parser.add_argument('-l', '--checkpoint_path', type=str)
parser.add_argument('-w', '--workers', type=int)
parser.add_argument('-st', '--train_sample_size', type=int)
parser.add_argument('-nv', '--no_validate',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('-sv', '--val_sample_size', type=int)
parser.add_argument('-bv', '--val_batch_size', type=int)
parser.add_argument('-ev', '--val_every_n_epochs', type=int)
parser.add_argument('-sd', '--seed', type=int)
args = parser.parse_args()

scm = args.scm
experiment_name = args.name or 'default_experiment'
density_estimator = 'gmm'
assert density_estimator in density_estimators
num_joint = args.num_joint or 1
assert num_joint in num_joints
context = args.context or ['e', 't', 'w_e', 'w_t']
mask = args.mask or ['mb', 'mb', 'mb', 'mb']
reduce = args.reduce or 'attn'
assert reduce in reduces
structures = args.structures or 10
hidden_feature = args.hidden_features or '128x2'
max_epoch = args.max_epochs or 200
checkpoint_path = args.checkpoint_path
workers = args.workers or 23
train_sample_size = args.train_sample_size or 1000
no_validate = args.no_validate or False
val_sample_size = args.val_sample_size or 1024
val_batch_size = args.val_batch_size or 32
val_every_n_epochs = args.val_every_n_epochs or max_epoch
seed = args.seed or 0

tmpl = \
    """
scm:
  scm_type: synthetic
  scm_kwargs:
    name: {scm}

evidence:
  evidence_type: context_masked
  evidence_kwargs:
    context_mode: 
{context}
    mask_mode:
{mask}
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
  model_type: ce
  model_kwargs:
    density_estimator_kwargs:
      {density_estimator_structures}
      hidden_features:
{hidden_features}
      reduce: {reduce}
    base_distribution_type: gaussian
    indicator_type: l1
    eval_sample_size: 1000
    train_sample_size: {train_sample_size}
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
    density_estimator_structures = {
        'gmm': f'components: {structures}',
    }[density_estimator]
    kwargs = {
        'scm': scm,
        'num_joint': num_joint,
        'reduce': reduce,
        'context': '\n'.join([' '*4 + '- ' + str(c) for c in context]),
        'mask': '\n'.join([' '*4 + '- ' + str(m) for m in mask]),
        'hidden_features': '\n'.join([' '*8 + '- ' + str(hdim) for hdim in hidden_features[hidden_feature]]),
        'density_estimator_structures': density_estimator_structures,
        'checkpoint_path': f'  checkpoint_path: {checkpoint_path}' if checkpoint_path is not None else '',
        'workers': workers,
        'max_epoch': max_epoch,
        'train_sample_size': train_sample_size,
        'val_check_epoch': max_epoch + 1 if no_validate else val_every_n_epochs,
        'val_sample_size': val_sample_size,
        'val_batch_size': val_batch_size,
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
    trial_name = f'j={num_joint},c={".".join(context)},k={".".join(mask)},r={reduce},t={structures},h={hidden_feature}'
    if not seed == 0:
        trial_name += f',{seed}'
    name = f'{experiment_name}/{scm}/{trial_name}'
    config_path = f'config/neural/{name}.yaml'
    log_path = f'output/{name}/logs'
    ckpt_path = f'output/{name}/checkpoints'

    # Skip complete trial
    res = get_train_result(ckpt_path)
    if isinstance(res, dict):
        epoch = res['epoch']
        if epoch == str(max_epoch-1):
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
    train_from_config_file(config_path, callback)


if __name__ == '__main__':
    trial()

"""
bash script/counterfactual/utils/sample/train_ce.sh\
    -n "test"\
    -s "simpson_nlin"\
    -j "5"\
    -c e+t w_e w_t -k mb1 mb1 em\
    -r attn -t 10 -d 64x2\
    -sv 1024 -bv 32 -ev 10\
    -e 200
"""
