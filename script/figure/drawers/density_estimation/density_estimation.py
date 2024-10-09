import os
import math
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import *
from tueplots import bundles, fontsizes
plt.rcParams.update(bundles.neurips2024())
fontsizes.neurips2024()

base_dir = 'output/density_estimation'
scms = ['simpson_nlin', 'triangle_nlin', 'largebd_nlin']
models = ['maf', 'nice']
methods = ['exom', 'causal_nf', 'rs', 'rs_causal_nf']
js = [1, 3, 5]
seeds = [0, 7, 42, 3407, 65535]
dims = {
    'simpson_nlin': 4,
    'triangle_nlin': 3,
    'largebd_nlin': 9,
}


def get_log(method, scm, model, j, seed, seed2):
    if method == 'exom':
        sub_dir = f'm={model},j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t=5,h=64x2'
    elif method == 'causal_nf':
        sub_dir = f'm={model},j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t=5,h=64x2'
    elif method == 'rs':
        sub_dir = f'j={j}'
    elif method == 'rs_causal_nf':
        sub_dir = f'j={j}'
    sub_dir = f'{method}/{scm}/' + sub_dir
    if seed > 0:
        sub_dir += f',{seed}'
    if seed2 > 0:
        sub_dir += f',cfs={seed2}'
    if not os.path.exists(os.path.join(base_dir, sub_dir, 'logs/val_logs.pt')):
        return False
    res = th.load(os.path.join(base_dir, sub_dir, 'logs/val_logs.pt'))
    return res


def get_estimate(log):
    last = list(log.keys())[-1]
    return th.cat([
        log[last][i]['estimate'] for i in range(len(log[last]))
    ], dim=-1).detach().cpu().numpy()


def df_csv_o(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    data = {
        'scm': [],
        'method': [],
        'model': [],
        'seed': [],
        'j': [],
        'id': [],
        'estimate': [],
    }

    def add_log(scm, method, model):
        for j in js:
            print(scm, method, model, j)
            for s in seeds:
                log = get_log(method, scm, model, j, s, 0)
                if not log:
                    continue
                log = get_estimate(log)
                for i, estimate in enumerate(log):
                    data['scm'].append(scm)
                    data['method'].append(method)
                    data['model'].append(model)
                    data['seed'].append(s)
                    data['j'].append(j)
                    data['id'].append(i)
                    data['estimate'].append(estimate)

    for scm in scms:
        for method in ['rs', 'exom']:
            if method == 'rs':
                add_log(scm, method, '-')
                continue
            for model in models:
                add_log(scm, method, model)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


dfo = df_csv_o(
    'script/figure/drawers/density_estimation/density_estimation_original.csv')


def df_csv_p(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    data = {
        'scm': [],
        'method': [],
        'model': [],
        'seed': [],
        'seed2': [],
        'j': [],
        'id': [],
        'estimate': [],
    }

    def add_log(scm, method, model):
        for j in js:
            print(scm, method, model, j)
            for s in seeds:
                for s2 in seeds:
                    log = get_log(method, scm, model, j, s, s2)
                    if not log:
                        continue
                    log = get_estimate(log)
                    for i, estimate in enumerate(log):
                        data['scm'].append(scm)
                        data['method'].append(method)
                        data['model'].append(model)
                        data['seed'].append(s)
                        data['seed2'].append(s2)
                        data['j'].append(j)
                        data['id'].append(i)
                        data['estimate'].append(estimate)

    for scm in scms:
        for method in ['rs_causal_nf', 'causal_nf']:
            if method == 'rs_causal_nf':
                add_log(scm, method, '-')
                continue
            for model in models:
                add_log(scm, method, model)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


dfp = df_csv_p(
    'script/figure/drawers/density_estimation/density_estimation_proxy.csv')


dim = {}
for scm in scms:
    dim[scm] = {}
    for j in js:
        val_dataset_path = f'script/counterfactual/density_estimation/val_saves/{scm}_{j}.pt'
        v = th.load(val_dataset_path)
        p = v['w_e_batched']
        p = p.reshape(-1, j * p.size(-1)).float().sum(dim=-1)
        dim[scm][j] = p.detach().cpu().numpy()


def select(df: pd.DataFrame, scm, j, method, model=None, seed2=None):
    if seed2 is None:
        if model is None:
            return df[(df['scm'] == scm) &
                      (df['j'] == j) &
                      (df['method'] == method)
                      ][['id', 'seed', 'estimate']]
        else:
            return df[(df['scm'] == scm) &
                      (df['j'] == j) &
                      (df['method'] == method) &
                      (df['model'] == model)
                      ][['id', 'seed', 'estimate']]
    else:
        if model is None:
            return df[(df['scm'] == scm) &
                      (df['j'] == j) &
                      (df['method'] == method) &
                      (df['seed2'] == seed2)
                      ][['id', 'seed', 'estimate']]
        else:
            return df[(df['scm'] == scm) &
                      (df['j'] == j) &
                      (df['method'] == method) &
                      (df['model'] == model) &
                      (df['seed2'] == seed2)
                      ][['id', 'seed', 'estimate']]


def to_exp_np(df: pd.DataFrame):
    X = df.sort_values(['id']).to_numpy()
    n = (X[:, 0].max().astype(int) + 1)
    X = X[:, -1].reshape(n, -1)
    return np.exp(X)


def ci_95(X: np.ndarray, dim: np.ndarray):  # ci from 5 trials, then averaged over 1024 cases
    Y = X.copy()
    for i in range(X.shape[-1]):
        # dimension regularizion
        # (which balances high dimensional testcases)
        Y[:, i] = X[:, i] ** (1/dim)
    mask = Y == 0
    Y = np.ma.masked_array(Y, mask)
    return 2 * (Y.std(axis=-1).mean(axis=0))


def zero(X: np.ndarray, dim: np.ndarray):  # 5 trials, then averaged over 1024 cases
    mask = X == 0
    # is equavalent to averaged over 5 \time 1024 cases
    return np.sum(mask) / mask.size


def table(scms=['simpson_nlin', 'triangle_nlin', 'largebd_nlin']):
    data = {
        'item': [],
        'scm': [],
        'j': [],
        'ci95': [],
        'zero': [],
    }

    def add_data(item, scm, j, ci95, zero):
        data['item'].append(item)
        data['scm'].append(scm)
        data['j'].append(j)
        data['ci95'].append(ci95)
        data['zero'].append(zero)

    def bias_o(model: str):
        for scm in scms:
            for j in js:
                if model == 'rs':
                    Y = select(dfo, scm, j, 'rs')
                else:
                    Y = select(dfo, scm, j, 'exom', model=model)
                x = ci_95(to_exp_np(Y), dim[scm][j])
                n = zero(to_exp_np(Y), dim[scm][j])
                add_data(f'{model}_o', scm, j, x, n)

    def bias_p(model: str):
        for scm in scms:
            for j in js:
                x1, n1 = [], []
                for s2 in seeds:
                    if model == 'rs':
                        Y = select(dfp, scm, j, 'rs_causal_nf', seed2=s2)
                    else:
                        Y = select(dfp, scm, j, 'causal_nf',
                                   model=model, seed2=s2)
                    if len(Y) > 0:
                        x1.append(ci_95(to_exp_np(Y), dim[scm][j]))
                        n1.append(zero(to_exp_np(Y), dim[scm][j]))
                if len(x1) == 0:
                    continue
                x = np.mean(x1)
                n = np.mean(n1)
                add_data(f'{model}_p', scm, j, x, n)

    bias_o('rs')
    bias_p('rs')
    bias_o('maf')
    bias_p('maf')
    bias_o('nice')
    bias_p('nice')

    return pd.DataFrame.from_dict(data)


table().to_csv('script/figure/drawers/density_estimation/density_stats.csv')


def to_float(cell, key):
    val = cell[[key]].to_numpy().item()
    try:
        val = float(val)
        if math.isinf(val) or str(val) == 'nan':
            return '--'
        return val
    except:
        return '--'


def select_j(df, method, i, j, scm):  # For density estimation
    cell = df[(df['item'] == f'{method}_{i}') &
              (df['j'] == j) &
              (df['scm'] == scm)]
    if len(cell) == 0:
        return '--', '--'
    return to_float(cell, 'zero'), to_float(cell, 'ci95')


def select_q(df, method, i, q, scm):  # For effect estimation
    cell = df[(df['item'] == f'{method}_{i}') &
              (df['q'] == q) &
              (df['scm'] == scm)]
    return to_float(cell, 'bias'), to_float(cell, 'ci95')


zero_ci95_fmt = '${:0.2f}_{{{:0.3f}}}$'
bias_ci95_fmt = '${:0.2f}_{{{:0.3f}}}$'


tmpl = """
\\label{{tab:2}}
\\centering
\\begin{{tabular}}{{ccccccccc}}
    \\toprule
    \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{SIMPSON-NLIN}} & \\multicolumn{{4}}{{c}}{{FAIRNESS}}\\\\
    \\cmidrule(r){{3-5}} \\cmidrule(r){{6-9}}
    Method & SCM & $|s|=1$ & $|s|=3$ & $|s|=5$ & ATE & ETT & NDE & CtfDE\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{RS}} & O & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[MAF]}} & O & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[NICE]}} & O & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {}\\\\
    \\bottomrule
\\end{{tabular}}
"""


def tab2():
    if not os.path.exists('script/figure/drawers/density_estimation/density_stats.csv'):
        return
    df1 = pd.read_csv(
        'script/figure/drawers/density_estimation/density_stats.csv')
    df2 = pd.read_csv(
        'script/figure/drawers/effect_estimation/effect_stats.csv')
    values = []
    for method in ['rs', 'maf', 'nice']:
        for i in ['o', 'p']:
            for j in [1, 3, 5]:
                scm = 'simpson_nlin'
                zero, ci95 = select_j(df1, method, i, j, scm)
                if ci95 == '--':
                    values.append('-')
                    continue
                values.append(zero_ci95_fmt.format(zero, ci95))
            for q in ['ate', 'ett', 'nde', 'ctfde']:
                scm = 'fairness'
                bias, ci95 = select_q(df2, method, i, q, scm)
                if ci95 == '--':
                    values.append('-')
                    continue
                values.append(bias_ci95_fmt.format(bias, ci95))
    tab2 = tmpl.format(*values)
    with open('script/figure/tabs/tab2.tex', 'w+', encoding='utf-8') as f:
        f.write(tab2)


tab2()


tmpl = """
\\label{{tab:91}}
\\centering
\\begin{{tabular}}{{cccccccc}}
    \\toprule
    \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{SIMPSON-NLIN}} & \\multicolumn{{2}}{{c}}{{TRIANGLE-NLIN}} & \\multicolumn{{1}}{{c}}{{LARGEBD-LIN}}\\\\
    \\cmidrule(r){{3-5}} \\cmidrule(r){{6-7}} \\cmidrule(r){{8-8}}
    Method & SCM & $|s|=1$ & $|s|=3$ & $|s|=5$ & $|s|=1$ & $|s|=3$ & $|s|=1$ \\\\
    \\midrule
    \\multirow{{2}}{{*}}{{RS}} & O & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[MAF]}} & O & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[NICE]}} & O & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {}\\\\
    \\bottomrule
\\end{{tabular}}
"""


def tabc91():
    df1 = pd.read_csv(
        'script/figure/drawers/density_estimation/density_stats.csv')
    values = []
    for method in ['rs', 'maf', 'nice']:
        for i in ['o', 'p']:
            for scm in ['simpson_nlin', 'triangle_nlin', 'largebd_nlin']:
                for j in [1, 3, 5]:
                    if scm == 'triangle_nlin' and j > 3:
                        continue
                    if scm == 'largebd_nlin' and j > 1:
                        continue
                    zero, ci95 = select_j(df1, method, i, j, scm)
                    if ci95 == '--':
                        values.append('-')
                        continue
                    values.append(zero_ci95_fmt.format(zero, ci95))
    tabc91 = tmpl.format(*values)
    with open('script/figure/tabs/tabc91.tex', 'w+', encoding='utf-8') as f:
        f.write(tabc91)


tabc91()
