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

base_dir = 'output/effect_estimation'
scms = ['fairness', 'fairness_xw', 'fairness_xy', 'fairness_yw']
models = ['maf', 'nice']
methods = ['exom', 'gan_ncm', 'rs', 'rs_gan_ncm',
           'rs_truth', 'rs_gan_ncm_truth']
qs = ['ate', 'ett', 'nde', 'ctfde']
seeds = [0, 7, 42, 3407, 65535]


def get_log(method, scm, model, q, seed, seed2):
    if method == 'exom':
        sub_dir = f'm={model},q={q},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t=5,h=64x2'
    elif method == 'gan_ncm':
        sub_dir = f'm={model},q={q},c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t=5,h=64x2'
    elif method == 'rs':
        sub_dir = f'q={q}'
    elif method == 'rs_gan_ncm':
        sub_dir = f'q={q}'
    elif method == 'rs_truth':
        sub_dir = f'q={q}'
    elif method == 'rs_gan_ncm_truth':
        sub_dir = f'q={q}'
    sub_dir = f'{method}/{scm}/' + sub_dir
    if seed > 0:
        sub_dir += f',{seed}'
    if seed2 > 0:
        sub_dir += f',cfs={seed2}'
    res = th.load(os.path.join(base_dir, sub_dir, 'logs/effect_logs.pt'))
    return res


def df_csv_o(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    data = {
        'scm': [],
        'method': [],
        'model': [],
        'seed': [],
        'q': [],
        'estimate': [],
    }

    def add_log(scm, method, model):
        for q in qs:
            for s in seeds:
                if method.endswith('truth') and s > 0:  # truth has only one trial
                    continue
                log = get_log(method, scm, model, q, s, 0)
                for estimate in log:
                    data['scm'].append(scm)
                    data['method'].append(method)
                    data['model'].append(model)
                    data['seed'].append(s)
                    data['q'].append(q)
                    data['estimate'].append(estimate)

    for scm in scms:
        for method in ['rs', 'rs_truth', 'exom']:
            if method == 'rs' or method == 'rs_truth':
                add_log(scm, method, '-')
                continue
            for model in models:
                add_log(scm, method, model)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


dfo = df_csv_o(
    'script/figure/drawers/effect_estimation/effect_estimation_original.csv')


def df_csv_p(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    data = {
        'scm': [],
        'method': [],
        'model': [],
        'seed': [],
        'seed2': [],
        'q': [],
        'estimate': [],
    }

    def add_log(scm, method, model):
        for q in qs:
            for s in seeds:
                if method.endswith('truth') and s > 0:  # truth has only one trial
                    continue
                for s2 in seeds:
                    log = get_log(method, scm, model, q, s, s2)
                    for estimate in log:
                        data['scm'].append(scm)
                        data['method'].append(method)
                        data['model'].append(model)
                        data['seed'].append(s)
                        data['seed2'].append(s2)
                        data['q'].append(q)
                        data['estimate'].append(estimate)

    for scm in scms:
        for method in ['rs_gan_ncm', 'rs_gan_ncm_truth', 'gan_ncm']:
            if method == 'rs_gan_ncm' or method == 'rs_gan_ncm_truth':
                add_log(scm, method, '-')
                continue
            for model in models:
                add_log(scm, method, model)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


dfp = df_csv_p(
    'script/figure/drawers/effect_estimation/effect_estimation_proxy.csv')


def select(df: pd.DataFrame, scm, q, method, model=None, seed2=None):
    if seed2 is None:
        if model is None:
            return df[(df['scm'] == scm) &
                      (df['q'] == q) &
                      (df['method'] == method)
                      ][['seed', 'estimate']]
        else:
            return df[(df['scm'] == scm) &
                      (df['q'] == q) &
                      (df['method'] == method) &
                      (df['model'] == model)
                      ][['seed', 'estimate']]
    else:
        if model is None:
            return df[(df['scm'] == scm) &
                      (df['q'] == q) &
                      (df['method'] == method) &
                      (df['seed2'] == seed2)
                      ][['seed', 'estimate']]
        else:
            return df[(df['scm'] == scm) &
                      (df['q'] == q) &
                      (df['method'] == method) &
                      (df['model'] == model) &
                      (df['seed2'] == seed2)
                      ][['seed', 'estimate']]


def to_np(df: pd.DataFrame):
    X = df.to_numpy()
    n = 1
    X = X[:, -1].reshape(n, -1)
    X = X[(X >= -1) & (X <= 1)].reshape(n, -1)  # remove abnormal
    return X


def ci_95(X: np.ndarray):   # ci from 5 trials, then averaged over 10 estimates
    return 2 * (X.std(axis=-1).mean(axis=0))


def bias(X: np.ndarray, mean):  # 5 trials \times 10 estimates
    # Note that in each trial each estimate, the estimand is the same
    # is equavalent to averaged over 5 \times 10 cases
    return np.mean(np.abs(X.reshape(-1) - mean))


def table(scms=['fairness', 'fairness_xw', 'fairness_xy', 'fairness_yw']):
    data = {
        'item': [],
        'scm': [],
        'q': [],
        'ci95': [],
        'bias': [],
    }

    def add_data(item, scm, q, ci95, bias):
        data['item'].append(item)
        data['scm'].append(scm)
        data['q'].append(q)
        data['ci95'].append(ci95)
        data['bias'].append(bias)

    def bias_o(model: str):
        for scm in scms:
            for q in qs:
                Y_truth = select(dfo, scm, q, 'rs_truth')
                Y_truth = Y_truth[['estimate']].mean().iloc[0]
                if model == 'rs':
                    Y = select(dfo, scm, q, 'rs')
                else:
                    Y = select(dfo, scm, q, 'exom', model=model)
                Y = to_np(Y)
                x = ci_95(Y)
                b = bias(Y, Y_truth)
                add_data(f'{model}_o', scm, q, x, b)

    def bias_p(model: str):
        for scm in scms:
            for q in qs:
                x1, b1 = [], []
                for s2 in seeds:
                    Y_truth = select(dfp, scm, q, 'rs_gan_ncm_truth', seed2=s2)
                    Y_truth = Y_truth[['estimate']].mean().iloc[0]
                    if model == 'rs':
                        Y = select(dfp, scm, q, 'rs_gan_ncm', seed2=s2)
                    else:
                        Y = select(dfp, scm, q, 'gan_ncm',
                                   model=model, seed2=s2)
                    if len(Y) > 0:
                        Y = to_np(Y)
                        x1.append(ci_95(Y))
                        b1.append(bias(Y, Y_truth))
                if len(x1) == 0:
                    continue
                x = np.mean(x1)  # mean over 5 proxy scms
                b = np.mean(b1)
                add_data(f'{model}_p', scm, q, x, b)

    bias_o('rs')
    bias_p('rs')
    bias_o('maf')
    bias_p('maf')
    bias_o('nice')
    bias_p('nice')

    return pd.DataFrame.from_dict(data)


table().to_csv('script/figure/drawers/effect_estimation/effect_stats.csv')


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
\\label{{tab:92}}
\\centering
\\begin{{tabular}}{{cccccccccc}}
    \\toprule
    \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{4}}{{c}}{{FAIRNESS}} & \\multicolumn{{4}}{{c}}{{FAIRNESS-XW}}\\\\
    \\cmidrule(r){{3-6}} \\cmidrule(r){{7-10}}
    Method & SCM & ATE & ETT & NDE & CtfDE & ATE & ETT & NDE & CtfDE\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{RS}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[MAF]}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[NICE]}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\bottomrule
\\end{{tabular}}
"""


def tabc92():
    df2 = pd.read_csv(
        'script/figure/drawers/effect_estimation/effect_stats.csv')
    values = []
    for method in ['rs', 'maf', 'nice']:
        for i in ['o', 'p']:
            for scm in ['fairness', 'fairness_xw']:
                for q in ['ate', 'ett', 'nde', 'ctfde']:
                    bias, ci95 = select_q(df2, method, i, q, scm)
                    if ci95 == '--':
                        values.append('-')
                        continue
                    values.append(bias_ci95_fmt.format(bias, ci95))
    tabc92 = tmpl.format(*values)
    with open('script/figure/tabs/tabc92.tex', 'w+', encoding='utf-8') as f:
        f.write(tabc92)


tabc92()


tmpl = """
\\label{{tab:93}}
\\centering
\\begin{{tabular}}{{cccccccccc}}
    \\toprule
    \\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{4}}{{c}}{{FAIRNESS-XY}} & \\multicolumn{{4}}{{c}}{{FAIRNESS-YW}}\\\\
    \\cmidrule(r){{3-6}} \\cmidrule(r){{7-10}}
    Method & SCM & ATE & ETT & NDE & CtfDE & ATE & ETT & NDE & CtfDE\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{RS}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[MAF]}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\midrule
    \\multirow{{2}}{{*}}{{EXOM[NICE]}} & O & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    & P & {} & {} & {} & {} & {} & {} & {} & {}\\\\
    \\bottomrule
\\end{{tabular}}
"""


def tabc93():
    df2 = pd.read_csv(
        'script/figure/drawers/effect_estimation/effect_stats.csv')
    values = []
    for method in ['rs', 'maf', 'nice']:
        for i in ['o', 'p']:
            for scm in ['fairness_xy', 'fairness_yw']:
                for q in ['ate', 'ett', 'nde', 'ctfde']:
                    bias, ci95 = select_q(df2, method, i, q, scm)
                    if ci95 == '--':
                        values.append('-')
                        continue
                    values.append(bias_ci95_fmt.format(bias, ci95))
    tabc93 = tmpl.format(*values)
    with open('script/figure/tabs/tabc93.tex', 'w+', encoding='utf-8') as f:
        f.write(tabc93)


tabc93()
