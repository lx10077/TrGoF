#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman" # Comment this if your computer doesn't support this font
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True, # Comment this if your computer doesn't support latex formula
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

alpha = 0.05

linestyles = ["-",":", "--", "-."]

color1 = 'tab:gray'
color2 = "black"
color3 = "white"

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

for k, use_mask in enumerate([True, False]):
    if use_mask is True:
        fig1_name = "7histogram/final-robust-1000(4-0.2-0.5w0.01None)-True"
        fig2_name = "7histogram/final-robust-1000(4-0.5-0.5w0.01None)-True"
    else:
        fig1_name = "7histogram/final-robust-1000(5-0.2-0.5w0.0031622776601683794None)-False"
        fig2_name = "7histogram/final-robust-1000(5-0.8-0.5w0.0031622776601683794None)-False"

    fig_names = [fig1_name, fig2_name]

    save_dict = json.load(open(fig_names[1]+".json", "r"))
    x1 = save_dict["H0"]

    n = 1000
    q = np.quantile(x1, 1-alpha)
    delta = q**2/np.log(np.log(n))-2
    print("delta", delta)
    print("q", q)

    x2 = save_dict["H1"]
    print("alpha (non-work)", np.sum(x2>=q)/n)

    save_dict = json.load(open(fig_names[0]+".json", "r"))
    x3 = save_dict["H1"]
    print("alpha (work)", np.sum(x3>=q)/n)

    n, bins, patches = ax[k,0].hist(x1, bins=90, density=False, facecolor = color1, edgecolor=color3, linewidth=0.5)
    for i in range(90):
        if bins[i] >= q:
            patches[i].set_fc(color2)
    ax[k,0].set_ylabel(r"Frequency")
    ax[k,0].text(0.95, 0.95, r"Under $H_0$" + "\n"+r'$\alpha=0.05$', transform=ax[k,0].transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='right')
    ax[k,0].set_xlim(xmax=20)

    n, bins, patches = ax[k,1].hist(x2, bins=90, density=False, facecolor = color1, edgecolor=color3, linewidth=0.5)
    for i in range(90):
        if bins[i] >= q:
            patches[i].set_fc(color2)

    if k == 0:
        ax[k,1].text(0.95, 0.95, r'Under $H_1$' +'\n'+ r'$(p, q)=(0.5,0.5)$'+ '\n' + r'$1-\beta=0.087$', transform=ax[k,1].transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right')
    else:
        ax[k,1].text(0.95, 0.95, r'Under $H_1$' +'\n'+ r'$(p, q)=(0.8,0.5)$'+ '\n' + r'$1-\beta=0.052$', transform=ax[k,1].transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right')
    ax[k,1].set_xlim(xmax=20)

    n_bins_here = 90 if use_mask else 200
    n, bins, patches = ax[k,2].hist(x3, bins=n_bins_here, density=False, facecolor = color1, edgecolor=color3, linewidth=0.5)
    for i in range(90):
        if bins[i] >= q:
            patches[i].set_fc(color2)

    if k == 0:
        ax[k,2].text(0.95, 0.95, r'Under $H_1$' +'\n'+ r'$(p, q)=(0.2,0.5)$'+ '\n' + r'$1-\beta=0.965$', 
                transform=ax[k,2].transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right')
    else:
        ax[k,2].text(0.95, 0.95, r'Under $H_1$' +'\n'+ r'$(p, q)=(0.2,0.5)$'+ '\n' + r'$1-\beta=1.0$', 
                transform=ax[k,2].transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='right')
 
ax[0,0].set_xlim(xmax=5)
ax[0,1].set_xlim(xmax=7)
ax[0,2].set_xlim(xmax=20)
ax[1,2].set_xlim(xmax=200)
plt.tight_layout()
plt.savefig(f'7histogram/robust-hist-both.pdf', dpi=300)
