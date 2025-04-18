#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"  # Comment this if your computer doesn't support this font
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True, # Comment this if your computer doesn't support latex formula
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def lower_q(K, n):
    low = np.log(K/(K-1))/np.log(n)
    print(low)
    return low

s_lst = [2, 1.5, 1, 0.5, 0]
fig, ax = plt.subplots(ncols=len(s_lst), nrows=3, figsize=(len(s_lst)*3+1, 9), sharey=True)

K = 5
N = 20

### For different n
q = 0.4
p = 0.3
N_trial = 1000
linestyles = ["-.", "-", "--", ":"]
ps = np.linspace(0.01, 1, 20)
qs = np.linspace(0.01, 1, 20)
for j, s in enumerate(s_lst):

    name = f'8contour/final-{p}-{q}-{K}-{N_trial}-{s}-p'
    save_dict =json.load(open(name+".json", 'r'))

    p2 = save_dict["p2"] 
    p3 = save_dict["p3"] 
    p4 = save_dict["p4"] 

    q2 = save_dict["q2"] 
    q3 = save_dict["q3"] 
    q4 = save_dict["q4"]

    ax[0,j].plot(ps, p2, label=r"$n=10^2$",linestyle=linestyles[0],color='black',)
    ax[0,j].plot(ps, p3, label=r"$n=10^3$",linestyle=linestyles[1],color='black',)
    ax[0,j].plot(ps, p4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
    ax[0,j].axvline(x=(1-q)/2, linestyle="dotted", color="red", linewidth=2)

    ax[0,j].set_xlabel(r"$p$")
    ax[0,j].set_title(rf"$s={s}$")
    ax[0,j].text(0.05, 0.95, rf'Fix $q=0.4$', transform=ax[0,j].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')

    ax[1,j].plot(ps, q2, label=r"$n=10^2$",linestyle=linestyles[0],color='black')
    ax[1,j].plot(ps, q3, label=r"$n=10^3$",linestyle=linestyles[1],color='black')
    ax[1,j].plot(qs, q4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
    ax[1,j].axvline(x=1-2*p, linestyle="dotted", color="red", linewidth=2)
    ax[1,j].text(0.05, 0.95, rf'Fix $p=0.3$', transform=ax[1,j].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')
    ax[1,j].set_xlabel(r"$q$")

    if j == len(s_lst)-1:
        ax[0,j].legend(frameon=False)
        ax[1,j].legend(frameon=False)
    if j == 0:
        ax[0,j].set_ylabel(r"Type I + II error")
        ax[1,j].set_ylabel(r"Type I + II error")

#### For countour

N=10
n_e = 4
n = 10**n_e
N_trial = 1000

ps = np.linspace(0.01, 1, N)
qs = np.linspace(lower_q(K, n), 1, N)

for j, s in enumerate(s_lst):

    name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-{s}"
    save_dict =json.load(open(name+".json", 'r'))
    Z = save_dict["Z"]

    CS0 = ax[2,j].contourf(ps, qs, Z, linewidths = 1, cmap=plt.cm.bone,vmax=1)
    x = np.linspace(0, 0.5, 100)
    ax[2,j].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)
    ax[2,j].set_xlabel(r"$p$")
    ax[2,j].set_xlim(xmin=0, xmax=1)
    ax[2,j].set_ylim(ymin=0, ymax=1)
    ax[2,j].clabel(CS0, inline=True, fontsize=10)

    if j == len(s_lst)-1:
        cbar = fig.colorbar(CS0,orientation="vertical")
    if j == 0:
        ax[2,j].set_ylabel(r"$q$")

plt.tight_layout()
plt.savefig(f'8contour/boundary-{N}-{N_trial}-{K}-{n}-others.pdf', dpi=300,bbox_inches='tight')
