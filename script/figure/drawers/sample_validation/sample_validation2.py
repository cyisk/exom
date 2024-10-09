import os
import shutil
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import *
from tueplots import bundles, fontsizes
plt.rcParams.update(bundles.neurips2024())
fontsizes.neurips2024()

base_dir = 'output/sample_validation'
methods = [
    'rs',
    'ce',
    'nis',
    'exom',
]
scms = [
    "chain_lin_3",
    "chain_nlin_3",
    "chain_lin_4",
    "chain_lin_5",
    "collider_lin",
    "fork_lin",
    "fork_nlin",
    "largebd_nlin",
    "simpson_nlin",
    "simpson_symprod",
    "triangle_lin",
    "triangle_nlin",
    "back_door",
    "front_door",
    "m",
    "napkin",
    "fairness",
    "fairness_xw",
    "fairness_xy",
    "fairness_yw",
]
scms2 = [
    "simpson_nlin",
    "napkin",
    "fairness_xw",
]
models = [
    "gmm",
    "maf",
    "nsf",
    "ncsf",
    "nice",
    "naf",
    "unaf",
    "sospf",
    "bpf",
]
js = [3]
seeds = [0, 7, 42, 3407, 65535]


def get_exom2_log(scm, model, j, seed):
    t = 10 if model == 'gmm' else 5
    sub_dir = f'exom2/{scm}/m={model},j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t={t},h=256x2'
    if seed > 0:
        sub_dir += f',{seed}'
    if not os.path.exists(os.path.join(base_dir, sub_dir, 'logs/val_logs.pt')):
        return False
    res = th.load(os.path.join(base_dir, sub_dir, 'logs/val_logs.pt'))
    resc = {}
    for epoch in res:
        resc[epoch] = {}
        keys = res[epoch][0].keys()
        for k in keys:
            resc[epoch][k] = th.cat([
                res[epoch][i][k] for i in range(len(res[epoch]))
            ], dim=-1)
    return resc


def mean(log):
    resc = {}
    for epoch in log:
        resc[epoch] = {}
        fail_mask = (log[epoch]['fails'] == 0)
        for k in log[epoch]:
            x = log[epoch][k]
            mask = ~(x.isnan() | x.isinf())
            if k not in ['effective_sample_proportion', 'fails']:
                mask &= fail_mask
            resc[epoch][k] = th.masked_select(
                x, mask
            ).mean().detach().cpu().item()
    return resc


def df_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    data = {
        'scm': [],
        'method': [],
        'model': [],
        'j': [],
        'seed': [],
        'ESP': [],
        'FR': [],
        'ESS': [],
        'ESE': [],
    }

    def add_log(log, scm, method, model, j, s):
        end = sorted(log.keys())[-1]
        data['scm'].append(scm)
        data['method'].append(method)
        data['model'].append(model)
        data['j'].append(j)
        data['seed'].append(s)
        data['ESP'].append(
            log[end]['effective_sample_proportion'])
        data['FR'].append(log[end]['fails'])
        data['ESS'].append(
            log[end]['effective_sample_size'])
        data['ESE'].append(
            log[end]['effective_sample_entropy'])

    def foreach_exom2():
        for scm in scms:
            for model in models:
                print(scm, 'exom', model)
                for j in js:
                    for s in seeds:
                        log = get_exom2_log(scm, model, j, s)
                        if log == False:
                            continue
                        log = mean(log)
                        add_log(log, scm, 'exom', model, j, s)

    foreach_exom2()

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)


df = df_csv('script/figure/drawers/sample_validation/sample_validation2.csv')


def fig82():
    path = 'script/figure/imgs/fig_app_c8_2'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    double_ = {}
    for key, value in fontsizes.neurips2024().items():
        double_[key] = 2.0 * value
    plt.rcParams.update(double_)
    fontsizes.neurips2024()

    for metric in ['ESP', 'FR']:
        df1 = df[(df['method'] == 'exom')][['scm', 'model', metric]]
        scms = df1['scm'].unique()
        models = df1['model'].unique()
        df1 = df1.groupby(['scm', 'model']).mean()
        res = {}
        for model in models:
            model_n = str.upper(model).replace('_', '-')
            res[model_n] = {}
            for scm in scms:
                scm_n = str.upper(scm).replace('_', '-')
                res[model_n][scm_n] = df1.loc[(scm, model), [
                    metric]].to_numpy().item()
        df2 = pd.DataFrame.from_dict(res)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        cmap = 'Greens' if metric == 'ESP' else 'Blues'
        sns.heatmap(df2, annot=True, ax=ax, cmap=cmap,
                    fmt='.2f', vmin=0, vmax=1)
        ax.set(xlabel='', ylabel='')
        plt.savefig(f'{path}/{metric}.pdf')


fig82()
