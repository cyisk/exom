import os
import shutil
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import *
from tueplots import bundles, fontsizes
plt.rcParams.update(bundles.neurips2024())
fontsizes.neurips2024()

base_dir = 'output/reduce_validation'
scms = ['simpson_nlin', 'largebd_nlin', 'triangle_nlin', 'm', 'napkin']
models = ['gmm', 'maf', 'nice', 'sospf']
js = [3, 5]
rs = ['concat', 'sum', 'wsum', 'attn']
seeds = [0, 7, 42, 3407, 65535]


def get_log(scm, model, j, r, seed):
    t = 10 if model == 'gmm' else 5
    sub_dir = f'{scm}/m={model},j={j},c=e+t.w_e.w_t,k=mb1.mb1.em,r={r},t={t},h=64x2'
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
        'model': [],
        'j': [],
        'reduce': [],
        'seed': [],
        'ESP': [],
        'FR': [],
        'ESS': [],
        'ESE': [],
    }
    for scm in scms:
        for model in models:
            print(scm, model)
            for j in js:
                for r in rs:
                    for s in seeds:
                        log = get_log(scm, model, j, r, s)
                        log = mean(log)
                        data['scm'].append(scm)
                        data['model'].append(model)
                        data['j'].append(j)
                        data['reduce'].append(r)
                        data['seed'].append(s)
                        data['ESP'].append(
                            log[199]['effective_sample_proportion'])
                        data['FR'].append(log[199]['fails'])
                        data['ESS'].append(log[199]['effective_sample_size'])
                        data['ESE'].append(
                            log[199]['effective_sample_entropy'])
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)


df = df_csv('script/figure/drawers/reduce_validation/reduce_validation.csv')


def fig_app_c7_3():
    path = 'script/figure/imgs/fig_app_c7_3'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    sns.set_style('whitegrid')
    double_ = {}
    for key, value in fontsizes.neurips2024().items():
        double_[key] = 2.0 * value
    plt.rcParams.update(double_)
    fontsizes.neurips2024()
    colors = sns.palettes.color_palette('colorblind6')
    df_plot = df.copy()

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'j': '$|s|$',
        'ESP': 'ESP',
    }
    df_plot = df_plot.replace(
        {'j': {1: '$|s|=1$', 3: '$|s|=3$', 5: '$|s|=5$'}})
    df_plot = df_plot.rename(columns=mapping)

    scms = ['m', 'napkin', 'simpson_nlin', 'largebd_nlin', 'triangle_nlin']
    models = ['gmm', 'maf', 'nice', 'sospf']

    def patch(ax):
        i = 0
        c = [colors[0], colors[1], colors[3], colors[2]]
        for p in ax.patches:
            if isinstance(p, PathPatch):
                p.set_facecolor(c[i // 2])
                i += 1

    # Separate subfigures
    for i, scm in enumerate(scms):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for j, model in enumerate(models):
            ax = axes[j]
            subdf = df_plot.copy()
            subdf = subdf[subdf['SCM'] == scm]
            subdf = subdf[subdf['Density Model'] == model]
            sns.boxplot(data=subdf, x="$|s|$", y="ESP",
                        hue="reduce", ax=ax,
                        flierprops={"marker": "x"})
            patch(ax)
            ax.get_legend().remove()
            ax.set(ylabel=None, xlabel=None)
            if i < 2:
                ax.set_xticklabels([])
        plt.savefig(f'{path}/{scm}.pdf')

    # Legend
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 1))
    handles = [
        Patch(facecolor=colors[0], edgecolor='k',
              label='\\rmfamily Concatenation'),
        Patch(facecolor=colors[1], edgecolor='k',
              label='\\rmfamily Summation'),
        Patch(facecolor=colors[3], edgecolor='k',
              label='\\rmfamily Weighted Summation'),
        Patch(facecolor=colors[2], edgecolor='k',
              label='\\rmfamily Attention'),
    ]
    L = ax.legend(handles=handles, ncols=2, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


def fig_app_c7_4():
    path = 'script/figure/imgs/fig_app_c7_4'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    sns.set_style('whitegrid')
    double_ = {}
    for key, value in fontsizes.neurips2024().items():
        double_[key] = 2.0 * value
    plt.rcParams.update(double_)
    fontsizes.neurips2024()
    colors = sns.palettes.color_palette('colorblind6')
    df_plot = df.copy()

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'j': '$|s|$',
        'FR': 'FR',
    }
    df_plot = df_plot.replace(
        {'j': {1: '$|s|=1$', 3: '$|s|=3$', 5: '$|s|=5$'}})
    df_plot = df_plot.rename(columns=mapping)

    scms = ['m', 'napkin', 'simpson_nlin', 'largebd_nlin', 'triangle_nlin']
    models = ['gmm', 'maf', 'nice', 'sospf']

    def patch(ax):
        c = [colors[0], colors[1], colors[3], colors[2]]
        ps = [p for p in ax.patches if isinstance(p, PathPatch)]
        lines_per_err = len(ax.lines) // len(ps)
        for j, l in enumerate(ax.lines):
            l.set_color(c[(j // lines_per_err) // 2])
        for j, p in enumerate(ps):
            p.set_facecolor('none')
            p.set_hatch('///')
            p.set_edgecolor(c[j // 2])
            p.set_linewidth(2)

    # Separate subfigures
    for i, scm in enumerate(scms):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for j, model in enumerate(models):
            ax = axes[j]
            subdf = df_plot.copy()
            subdf = subdf[subdf['SCM'] == scm]
            subdf = subdf[subdf['Density Model'] == model]
            sns.boxplot(data=subdf, x="$|s|$", y="FR",
                        hue="reduce", ax=ax,
                        flierprops={"marker": "x"})
            patch(ax)
            ax.get_legend().remove()
            ax.set(ylabel=None, xlabel=None)
            if i < 2:
                ax.set_xticklabels([])
        plt.savefig(f'{path}/{scm}.pdf')

    # Legend
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 1))
    handles = [
        Patch(facecolor='none', edgecolor=colors[0], linewidth=2, hatch='///',
              label='\\rmfamily Concatenation'),
        Patch(facecolor='none', edgecolor=colors[1], linewidth=2, hatch='///',
              label='\\rmfamily Summation'),
        Patch(facecolor='none', edgecolor=colors[3], linewidth=2, hatch='///',
              label='\\rmfamily Weighted Summation'),
        Patch(facecolor='none', edgecolor=colors[2], linewidth=2, hatch='///',
              label='\\rmfamily Attention'),
    ]
    L = ax.legend(handles=handles, ncols=2, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


fig_app_c7_3()
fig_app_c7_4()
