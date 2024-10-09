from typing import *
import argparse
import os
import re
import shutil
import torch as th
from tqdm import tqdm

from model.counterfactual.query import *


# Preset candidates
queries = ['ate', 'ett', 'nde', 'ctfde']
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
parser.add_argument('-q', '--query', type=str)
parser.add_argument('-w', '--workers', type=int)
parser.add_argument('-nv', '--no_validate',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('-sv', '--val_sample_size', type=int)
parser.add_argument('-ev', '--eval_sample_size', type=int)
parser.add_argument('-bv', '--val_batch_size', type=int)
parser.add_argument('-sd', '--seed', type=int)
parser.add_argument('-pd', '--gan_ncm_hidden_features', type=str)
parser.add_argument('-ps', '--gan_ncm_seed', type=int)
args = parser.parse_args()

scm = args.scm
experiment_name = args.name or 'default_experiment'
query = args.query or 'ate'
assert query in queries
workers = args.workers or 23
no_validate = args.no_validate or False
val_sample_size = args.val_sample_size or 1024
eval_sample_size = args.eval_sample_size or 10000
val_batch_size = args.val_batch_size or 32
seed = args.seed or 0
gan_ncm_hidden_features = args.gan_ncm_hidden_features or '64x2'
gan_ncm_seed = args.gan_ncm_seed or 0

query_kwstr = """
        Y: y
        X: x
        x0: 0
        x1: 1
"""
query_kwargs = dict(Y='y', y1=1, X='x', x0=0, x1=1)
if query in ['nde', 'ctfde']:
    query_kwstr += """
        W: w
        w:
        - 0
        - 1
"""
    query_kwargs.update(dict(W='w', w=[0, 1]))
num_joint = {
    'ate': 1,
    'ett': 2,
    'nde': 2,
    'ctfde': 3,
}[query]


tmpl = \
    """
scm:
  scm_type: proxy
  scm_kwargs:
    proxy_type: gan_ncm
    base_scm_kwargs:
      name: {scm}
    model_kwargs:
      ncm_hidden_features:
{gan_ncm_hidden_features}
      critic_hidden_features:
{gan_ncm_hidden_features}
      exogenous_distribution_type: gaussian
      learning_rate: 0.001
    checkpoint_path: {gan_ncm_checkpoint_path}

evidence:
  evidence_type: context_concat
  evidence_kwargs:
    context_mode: 
    - e
  batched: true
  max_len_joint: {num_joint}

train_dataset:
  sampler:
    sampler_type: query
    sampler_kwargs:
      query_type: {query}
      query_kwargs:
{query_kwstr}
  size: 16384

train_dataloader:
  batch_size: 256
  num_workers: {workers}

val_dataset:
  sampler:
    sampler_type: query
    sampler_kwargs:
      query_type: {query}
      query_kwargs:
{query_kwstr}
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
    gan_ncm_checkpoint_path: str,
    name: str = 'default_experiment',
):
    kwargs = {
        'scm': scm,
        'num_joint': num_joint,
        'query': query,
        'query_kwstr': query_kwstr,
        'workers': workers,
        'val_sample_size': val_sample_size,
        'val_batch_size': val_batch_size,
        'eval_sample_size': eval_sample_size,
        'name': name,
        'seed': seed,
        'gan_ncm_hidden_features': '\n'.join([' '*8 + '- ' + str(hdim) for hdim in hidden_features[gan_ncm_hidden_features]]),
        'gan_ncm_checkpoint_path': gan_ncm_checkpoint_path,
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
    trial_name = f'q={query}'
    if not seed == 0:
        trial_name += f',{seed}'
    if not gan_ncm_seed == 0:
        trial_name += f',cfs={gan_ncm_seed}'
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

    # Automatic find checkpoint
    gan_ncm_name = f'h={gan_ncm_hidden_features}'
    if not gan_ncm_seed == 0:
        gan_ncm_name += f',{gan_ncm_seed}'
    gan_ncm_checkpoint_path = f'output/proxy_scm/gan_ncm/{scm}/{gan_ncm_name}/checkpoints/'
    if not os.path.exists(gan_ncm_checkpoint_path):
        return False
    if len(os.listdir(gan_ncm_checkpoint_path)) == 0:
        return False
    ckpt_name = os.listdir(gan_ncm_checkpoint_path)[0]
    gan_ncm_checkpoint_path = os.path.join(
        gan_ncm_checkpoint_path, ckpt_name
    )

    # Generate configuration
    config = config_tmpl(gan_ncm_checkpoint_path, name)
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
    outputs = train_from_config_file(config_path, callback)
    evidence_type = outputs['evidence_type']
    evidence_kwargs = outputs['evidence_kwargs']
    exom = outputs['model'].to('cuda:0')

    # Query
    Q = {
        'ate': ATE,
        'ett': ETT,
        'nde': NDE,
        'ctfde': CtfDE,
    }[query]
    res = []
    for i in tqdm(range(10)):
        q = Q(
            estimator=exom,
            **query_kwargs,
            evidence_type=evidence_type,
            evidence_kwargs=evidence_kwargs,
        )
        res.append(q)
    print(res)
    test_log_path = os.path.join(log_path, 'effect_logs.pt')
    return th.save(res, test_log_path)


if __name__ == '__main__':
    trial()

"""
bash script/counterfactual/utils/effect/train_exom_from_gan_ncm.sh\
    -n "test_exom_from_gan_ncm"\
    -pd 64x2\
    -s "fairness"\
    -q "ate"\
    -sv 1024 -bv 32
"""
