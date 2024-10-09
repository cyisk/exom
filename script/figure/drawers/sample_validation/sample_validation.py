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
js = [1, 3, 5]
seeds = [0, 7, 42, 3407, 65535]


def get_exom_log(scm, model, j, seed):
    t = 10 if model == 'gmm' else 5
    sub_dir = f'exom/{scm}/m={model},j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t={t},h=64x2'
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


def get_rs_log(scm, j, seed):
    sub_dir = f'rs/{scm}/j={j}'
    if seed > 0:
        sub_dir += f',{seed}'
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


def get_ce_log(scm, j, seed):
    t = 10
    sub_dir = f'ce/{scm}/j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t={t},h=64x2'
    if seed > 0:
        sub_dir += f',{seed}'
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


def get_nis_log(scm, j, seed):
    t = 5
    sub_dir = f'nis/{scm}/j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t={t},h=64x2'
    if seed > 0:
        sub_dir += f',{seed}'
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

    def foreach_exom():
        for scm in scms:
            for model in models:
                print(scm, 'exom', model)
                for j in js:
                    for s in seeds:
                        log = get_exom_log(scm, model, j, s)
                        if log == False:
                            continue
                        log = mean(log)
                        add_log(log, scm, 'exom', model, j, s)

    def foreach_rs():
        for scm in scms:
            print(scm, 'rs')
            for j in js:
                for s in seeds:
                    log = get_rs_log(scm, j, s)
                    log = mean(log)
                    add_log(log, scm, 'rs', '-', j, s)

    def foreach_ce():
        for scm in scms2:
            print(scm, 'ce')
            for j in js:
                for s in seeds:
                    log = get_ce_log(scm, j, s)
                    log = mean(log)
                    add_log(log, scm, 'ce', '-', j, s)

    def foreach_nis():
        for scm in scms2:
            print(scm, 'nis')
            for j in js:
                for s in seeds:
                    log = get_nis_log(scm, j, s)
                    log = mean(log)
                    add_log(log, scm, 'nis', '-', j, s)

    foreach_exom()
    foreach_rs()
    foreach_ce()
    foreach_nis()

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)


df = df_csv('script/figure/drawers/sample_validation/sample_validation.csv')


tmpl = """
\\label{{tab:1}}
\\centering
\\begin{{tabular}}{{cccccccc}}
    \\toprule
    \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{2}}{{c}}{{SIMPSON-NLIN}} & \\multicolumn{{2}}{{c}}{{NAPKIN}} & \\multicolumn{{2}}{{c}}{{FAIRNESS-XW}}\\\\
    \\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
    $|s|$ & Model & \\multicolumn{{1}}{{c}}{{ESP}} & \\multicolumn{{1}}{{c}}{{FR}} & \\multicolumn{{1}}{{c}}{{ESP}} & \\multicolumn{{1}}{{c}}{{FR}} & \\multicolumn{{1}}{{c}}{{ESP}} & \\multicolumn{{1}}{{c}}{{FR}}\\\\
    \\midrule
    \\multirow{{5}}{{*}}{{1}} & RS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & CEIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & NIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[GMM] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[MAF] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    \\midrule
    \\multirow{{5}}{{*}}{{3}} & RS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & CEIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & NIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[GMM] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[MAF] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    \\midrule
    \\multirow{{5}}{{*}}{{5}} & RS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & CEIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & NIS & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[GMM] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    & EXOM[MAF] & ${}$ & ${}$ & ${}$ & ${}$ & ${}$ & ${}$\\\\
    \\bottomrule
\\end{{tabular}}
"""


def tab1():
    values = []
    matrix = np.zeros((15, 6))
    for i1, j in enumerate([1, 3, 5]):
        for i2, method in enumerate(['rs', 'ce', 'nis', 'gmm', 'maf']):
            for j1, scm in enumerate(['simpson_nlin', 'napkin', 'fairness_xw']):
                for j2, metric in enumerate(['ESP', 'FR']):
                    if method in ['rs', 'ce', 'nis']:
                        value = df[
                            (df['method'] == method) &
                            (df['scm'] == scm) &
                            (df['j'] == j)
                        ][[metric]]
                        assert len(value) == 5
                    else:
                        value = df[
                            (df['method'] == 'exom') &
                            (df['model'] == method) &
                            (df['scm'] == scm) &
                            (df['j'] == j)
                        ][[metric]]
                    value = value.to_numpy().mean().item()
                    values.append('{:.3f}'.format(value))
                    matrix[i1 * 5 + i2, j1 * 2 + j2] = value
    for j in range(6):
        for i in range(3):
            if j % 2 == 0:
                k = np.argmax(matrix[(i*5):(i+1)*5, j])
            else:
                k = np.argmin(matrix[(i*5):(i+1)*5, j])
            ijk = (i * 5 + k) * 6 + j
            values[ijk] = f'\mathbf{{{values[ijk]}}}'
    tab1 = tmpl.format(*values)
    with open('script/figure/tabs/tab1.tex', 'w+', encoding='utf-8') as f:
        f.write(tab1)


tab1()


def fig8():
    path = 'script/figure/imgs/fig_app_c8'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    double_ = {}
    for key, value in fontsizes.neurips2024().items():
        double_[key] = 2.0 * value
    plt.rcParams.update(double_)
    fontsizes.neurips2024()

    for metric in ['ESP', 'FR']:
        df1 = df[(df['j'] == 3) & (df['method'] == 'exom')
                 ][['scm', 'model', metric]]
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


fig8()
