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

base_dir = 'output/mask_validation'
scms = ['simpson_nlin', 'largebd_nlin', 'triangle_nlin', 'm', 'napkin']
models = ['gmm', 'maf', 'nice', 'sospf']
js = [1, 3, 5]
ks = ['em.em.em', 'mb.mb.em', 'mb1.mb1.em', 'mb2.mb2.em']
seeds = [0, 7, 42, 3407, 65535]


def get_log(scm, model, j, k, seed):
    t = 10 if model == 'gmm' else 5
    h = '256x2' if model in ['nice', 'sospf'] else '64x2'
    sub_dir = f'{scm}/m={model},j={j},c=e+t.w_e.w_t,k={k},r=attn,t={t},h={h}'
    if not os.path.exists(os.path.join(base_dir, sub_dir, 'logs/val_logs.pt')):
        h = '64x2'
        sub_dir = f'{scm}/m={model},j={j},c=e+t.w_e.w_t,k={k},r=attn,t={t},h={h}'
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
        'mask': [],
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
                for k in ks:
                    for s in seeds:
                        log = get_log(scm, model, j, k, s)
                        log = mean(log)
                        data['scm'].append(scm)
                        data['model'].append(model)
                        data['j'].append(j)
                        data['mask'].append(k)
                        data['seed'].append(s)
                        data['ESP'].append(
                            log[199]['effective_sample_proportion'])
                        data['FR'].append(log[199]['fails'])
                        data['ESS'].append(log[199]['effective_sample_size'])
                        data['ESE'].append(
                            log[199]['effective_sample_entropy'])
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


df = df_csv('script/figure/drawers/mask_validation/mask_validation.csv')


def fig4():
    path = 'script/figure/imgs/fig4'
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
    df_plot = df[df['mask'].isin(['em.em.em', 'mb1.mb1.em'])]

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'j': '$|s|$',
        'ESP': 'ESP',
    }
    df_plot = df_plot.replace(
        {'j': {1: '$|s|=1$', 3: '$|s|=3$', 5: '$|s|=5$'}})
    df_plot = df_plot.rename(columns=mapping)

    scms = ['simpson_nlin', 'largebd_nlin', 'm', 'napkin']
    models = ['gmm', 'maf', 'nice', 'sospf']

    def patch(ax):
        i = 0
        c = [colors[0], colors[2]]
        for p in ax.patches:
            if isinstance(p, PathPatch):
                p.set_facecolor(c[i // 3])
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
                        hue="mask", ax=ax,
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
              label='\\rmfamily without Markov Boundary Masked'),
        Patch(facecolor=colors[2], edgecolor='k',
              label='\\rmfamily Markov Boundary Masked'),
    ]
    L = ax.legend(handles=handles, ncols=2, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


def fig_app_c7():
    path = 'script/figure/imgs/fig_app_c7'
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
        c = [colors[0], colors[1], colors[2], colors[3]]
        for p in ax.patches:
            if isinstance(p, PathPatch):
                p.set_facecolor(c[i // 3])
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
                        hue="mask", ax=ax,
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
              label='\\rmfamily without Markov Boundary Masked'),
        Patch(facecolor=colors[1], edgecolor='k',
              label='\\rmfamily Markov Boundary Masked (All Cut)'),
        Patch(facecolor=colors[2], edgecolor='k',
              label='\\rmfamily Markov Boundary Masked (Endo. Cut)'),
        Patch(facecolor=colors[3], edgecolor='k',
              label='\\rmfamily Markov Boundary Masked (No Cut)'),
    ]
    L = ax.legend(handles=handles, ncols=2, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


def fig_app_c7_2():
    path = 'script/figure/imgs/fig_app_c7_2'
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
        c = [colors[0], colors[1], colors[2], colors[3]]
        ps = [p for p in ax.patches if isinstance(p, PathPatch)]
        lines_per_err = len(ax.lines) // len(ps)
        for j, l in enumerate(ax.lines):
            l.set_color(c[(j // lines_per_err) // 3])
        for j, p in enumerate(ps):
            p.set_facecolor('none')
            p.set_hatch('///')
            p.set_edgecolor(c[j // 3])
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
                        hue="mask", ax=ax,
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
              label='\\rmfamily without Markov Boundary Masked'),
        Patch(facecolor='none', edgecolor=colors[1], linewidth=2, hatch='///',
              label='\\rmfamily Markov Boundary Masked (All Cut)'),
        Patch(facecolor='none', edgecolor=colors[2], linewidth=2, hatch='///',
              label='\\rmfamily Markov Boundary Masked (Endo. Cut)'),
        Patch(facecolor='none', edgecolor=colors[3], linewidth=2, hatch='///',
              label='\\rmfamily Markov Boundary Masked (No Cut)'),
    ]
    L = ax.legend(handles=handles, ncols=2, frameon=False)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


fig4()
fig_app_c7()
fig_app_c7_2()
