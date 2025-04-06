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

# Figure 1

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(13, 3),sharey='row',sharex='col',)

K = 5
N = 20
n_e = 4
n = 10**n_e
N_trial = 100

ps = np.linspace(0.01, 1, N)
qs = np.linspace(lower_q(K, n), 1, N)

name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-ars"
save_dict =json.load(open(name+".json", 'r'))
ars_z = save_dict["Z"]

name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-log"
save_dict =json.load(open( name+".json", 'r'))
log_z = save_dict["Z"]

name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-ind05"
save_dict =json.load(open( name+".json", 'r'))
ind05_z = save_dict["Z"]

name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-ind08"
save_dict =json.load(open( name+".json", 'r'))
ind08_z = save_dict["Z"]

name = f"8contour/contour-{N}-{N_trial}-{K}-{n}-opt01"
save_dict =json.load(open( name+".json", 'r'))
opt01_z = save_dict["Z"]

baseline = "yellow"

CS0 = ax[0].contourf(ps, qs, ars_z, linewidths = 1,cmap=plt.cm.bone,vmax=1)
x = np.linspace(0, 0.5, 100)
ax[0].plot(x, 0.5-x, linestyle="dotted", color=baseline, linewidth=2)
ax[0].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)

ax[0].set_xlabel(r"$p$")
ax[0].set_ylabel(r"$q$")
ax[0].set_title(r"$h_{\mathrm{ars}}$")
ax[0].set_aspect('equal')
ax[0].clabel(CS0, inline=True, fontsize=10)
ax[0].set_xlim(xmin=0, xmax=1)
ax[0].set_ylim(ymin=0, ymax=1)

CS1 = ax[1].contourf(ps, qs, log_z, linewidths = 1, cmap=plt.cm.bone, vmax=1)
x = np.linspace(0, 0.5, 100)
ax[1].plot(x, 0.5-x, linestyle="dotted", color=baseline, linewidth=2)
ax[1].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)

ax[1].set_xlabel(r"$p$")
ax[1].set_title(r"$h_{\mathrm{log}}$")
ax[1].set_aspect('equal')
ax[1].clabel(CS1, inline=True, fontsize=10)
ax[1].set_xlim(xmin=0, xmax=1)
ax[1].set_ylim(ymin=0, ymax=1)

CS2 = ax[2].contourf(ps, qs, ind05_z, linewidths = 1, cmap=plt.cm.bone, vmax=1)
x = np.linspace(0, 0.5, 100)
ax[2].plot(x, 0.5-x, linestyle="dotted", color=baseline, linewidth=2)
ax[2].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)

ax[2].set_xlabel(r"$p$")
ax[2].set_aspect('equal')
ax[2].set_title(r"$h_{\mathrm{ind}, 0.5}$")
ax[2].clabel(CS2, inline=True, fontsize=10)
ax[2].set_xlim(xmin=0, xmax=1)
ax[2].set_ylim(ymin=0, ymax=1)

CS3 = ax[3].contourf(ps, qs, opt01_z, linewidths = 1, cmap=plt.cm.bone, vmax=1)
x = np.linspace(0, 0.5, 100)
ax[3].plot(x, 0.5-x, linestyle="dotted", color=baseline, linewidth=2)
ax[3].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)

ax[3].set_xlabel(r"$p$")
ax[3].set_aspect('equal')
ax[3].set_title(r"$h_{\mathrm{opt}, 0.1}$")
ax[3].clabel(CS3, inline=True, fontsize=10)
ax[3].set_xlim(xmin=0, xmax=1)
ax[3].set_ylim(ymin=0, ymax=1)

plt.colorbar(CS3, ax=ax)
plt.savefig(f'8contour/contour-{N}-{N_trial}-{K}-{n}-simple.pdf', dpi=300,bbox_inches='tight')
