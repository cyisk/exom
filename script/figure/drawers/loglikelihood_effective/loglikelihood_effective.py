import os
import shutil
import torch as th
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import *
from matplotlib.lines import Line2D
from tueplots import bundles, fontsizes
plt.rcParams.update(bundles.neurips2024())
fontsizes.neurips2024()

base_dir = 'output/loglikelihood_effective'
scms = ['simpson_nlin', 'largebd_nlin', 'napkin', 'fairness_xw']
models = ['gmm', 'maf', 'nice', 'sospf']
seeds = [0, 7, 42, 3407, 65535]


def get_log(scm, model, seed):
    t = 10 if model == 'gmm' else 5
    h = '256x2' if model in ['nice', 'sospf'] else '64x2'
    sub_dir = f'{scm}/m={model},j=3,c=e+t.w_e.w_t,k=mb1.mb1.em,r=attn,t={t},h={h}'
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
            if k in ['log_likelihood']:
                resc[epoch][k] = th.masked_select(
                    x, ~(x.isnan() | x.isinf())
                ).median().detach().cpu().item()
                continue
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
        'seed': [],
        'epoch': [],
        'ESP': [],
        'FR': [],
        'ESS': [],
        'ESE': [],
        'LL': [],
    }
    for scm in scms:
        for model in models:
            print(scm, model)
            for s in seeds:
                log = get_log(scm, model, s)
                log = mean(log)
                for epoch in log:
                    data['scm'].append(scm)
                    data['model'].append(model)
                    data['seed'].append(s)
                    data['epoch'].append(epoch)
                    data['ESP'].append(
                        log[epoch]['effective_sample_proportion'])
                    data['FR'].append(log[epoch]['fails'])
                    data['ESS'].append(log[epoch]['effective_sample_size'])
                    data['ESE'].append(
                        log[epoch]['effective_sample_entropy'])
                    data['LL'].append(
                        log[epoch]['log_likelihood'])
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path)
    return df


df = df_csv(
    'script/figure/drawers/loglikelihood_effective/loglikelihood_effective.csv'
)


def fig3():
    path = 'script/figure/imgs/fig3'
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

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'ESP': '\\rmfamily ESP',
        'LL': '\\rmfamily LL',
        'FR': '\\rmfamily FR',
    }
    df_plot = df.rename(columns=mapping)

    def patch(ax):
        for i, l in enumerate(ax.lines):
            x = l.get_xdata()
            y = l.get_ydata()
            if len(y) > 0:
                ax.plot(x[:1], y[:1], color=colors[i],
                        marker='o', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])
                ax.plot(x[-1:], y[-1:], color=colors[i],
                        marker='X', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])

    # Separate subfigures
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    scm = 'simpson_nlin'
    model = 'maf'
    for i, metric in enumerate(['\\rmfamily ESP', '\\rmfamily FR']):
        ax = axes[i]
        subdf = df_plot.copy()
        subdf = subdf[subdf['SCM'] == scm]
        subdf = subdf[subdf['Density Model'] == model]
        subdf = subdf[['epoch', 'seed',
                       '\\rmfamily ESP', '\\rmfamily FR', '\\rmfamily LL']]
        subdf['\\rmfamily FR'] = subdf['\\rmfamily FR'].fillna(0)
        subdf.dropna()
        subdf = subdf.sort_values(by=['epoch'])
        sns.lineplot(
            data=subdf, y=metric, x="\\rmfamily LL",
            hue='seed', ax=ax, sort=False, palette='colorblind6',
            linestyle='--',
        )
        patch(ax)
        ax.get_legend().remove()
    plt.savefig(f'{path}/ll_esp_fr.pdf')

    # Legend
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))
    handles = [
        Line2D(
            [],
            [],
            lw=1.5,
            linestyle="--",
            color="k",
            label="\\rmfamily One Run",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='o', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Initial State",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='X', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Final State",
        ),
    ]
    L = ax.legend(handles=handles, ncols=3, frameon=False)
    ax.axis("off")
    plt.setp(L.texts, family='Times New Roman')
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


def fig_app_c5():
    path = 'script/figure/imgs/fig_app_c5'
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

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'ESP': '\\rmfamily ESP',
        'LL': '\\rmfamily LL',
    }
    df_plot = df.rename(columns=mapping)

    def patch(ax):
        for i, l in enumerate(ax.lines):
            x = l.get_xdata()
            y = l.get_ydata()
            if len(y) > 0:
                ax.plot(x[:1], y[:1], color=colors[i],
                        marker='o', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])
                ax.plot(x[-1:], y[-1:], color=colors[i],
                        marker='X', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])

    # Separate subfigures
    for i, scm in enumerate(scms):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for j, model in enumerate(models):
            ax = axes[j]
            subdf = df_plot.copy()
            subdf = subdf[subdf['SCM'] == scm]
            subdf = subdf[subdf['Density Model'] == model]
            subdf = subdf[['epoch', 'seed',
                           '\\rmfamily ESP', '\\rmfamily LL']]
            subdf.dropna()
            subdf = subdf.sort_values(by=['epoch'])
            sns.lineplot(
                data=subdf, y="\\rmfamily ESP", x="\\rmfamily LL",
                hue='seed', ax=ax, sort=False, palette='colorblind6',
                linestyle='--',
            )
            patch(ax)
            ax.get_legend().remove()
            ax.set(ylabel=None, xlabel=None)
        plt.savefig(f'{path}/{scm}.pdf')

    # Legend
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))
    handles = [
        Line2D(
            [],
            [],
            lw=1.5,
            linestyle="--",
            color="k",
            label="\\rmfamily One Run",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='o', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Initial State",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='X', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Final State",
        ),
    ]
    L = ax.legend(handles=handles, ncols=3, frameon=False)
    ax.axis("off")
    plt.setp(L.texts, family='Times New Roman')
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


def fig_app_c5_2():
    path = 'script/figure/imgs/fig_app_c5_2'
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

    mapping = {
        'scm': 'SCM',
        'model': 'Density Model',
        'FR': '\\rmfamily FR',
        'LL': '\\rmfamily LL',
    }
    df_plot = df.rename(columns=mapping)

    def patch(ax):
        for i, l in enumerate(ax.lines):
            x = l.get_xdata()
            y = l.get_ydata()
            if len(y) > 0:
                ax.plot(x[:1], y[:1], color=colors[i],
                        marker='o', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])
                ax.plot(x[-1:], y[-1:], color=colors[i],
                        marker='X', markersize=10,
                        markerfacecolor=(*colors[i], 0.5), markeredgecolor=colors[i])

    # Separate subfigures
    for i, scm in enumerate(scms):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for j, model in enumerate(models):
            ax = axes[j]
            subdf = df_plot.copy()
            subdf = subdf[subdf['SCM'] == scm]
            subdf = subdf[subdf['Density Model'] == model]
            subdf = subdf[['epoch', 'seed',
                           '\\rmfamily FR', '\\rmfamily LL']]
            subdf['\\rmfamily FR'] = subdf['\\rmfamily FR'].fillna(1)
            subdf.dropna()
            subdf = subdf.sort_values(by=['epoch'])
            sns.lineplot(
                data=subdf, y="\\rmfamily FR", x="\\rmfamily LL",
                hue='seed', ax=ax, sort=False, palette='colorblind6',
                linestyle='--',
            )
            patch(ax)
            ax.get_legend().remove()
            ax.set(ylabel=None, xlabel=None)
        plt.savefig(f'{path}/{scm}.pdf')

    # Legend
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))
    handles = [
        Line2D(
            [],
            [],
            lw=1.5,
            linestyle="--",
            color="k",
            label="\\rmfamily One Run",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='o', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Initial State",
        ),
        Line2D(
            [],
            [],
            linestyle='None',
            color='k',
            marker='X', markersize=10,
            markerfacecolor=(0, 0, 0, 0.5), markeredgecolor='k',
            label="\\rmfamily Final State",
        ),
    ]
    L = ax.legend(handles=handles, ncols=3, frameon=False)
    ax.axis("off")
    plt.setp(L.texts, family='Times New Roman')
    plt.tight_layout()
    plt.savefig(f'{path}/legend.pdf')


fig3()
fig_app_c5()
fig_app_c5_2()
