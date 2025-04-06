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

K = 5
search = 21
N_trial = 1000

def lower_q(K, n):
    low = np.log(K/(K-1))/np.log(n)
    print(low)
    return low


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

q = 0.4
p = 0.3
ps = np.linspace(0.01, 1, 20)
qs = np.linspace(0.01, 1, 20)

for k, use_mask in enumerate([True, False]):
    if use_mask is True:
        name = f'8contour/final-{p}-{q}-{K}-{N_trial}-2-p'
    else:
        name = f'8contour/final-{p}-{q}-{K}-{N_trial}-2-p-nomask'

    save_dict =json.load(open( name+".json", 'r'))

    p2 = save_dict["p2"] 
    p3 = save_dict["p3"] 
    p4 = save_dict["p4"] 

    q2 = save_dict["q2"] 
    q3 = save_dict["q3"] 
    q4 = save_dict["q4"]

    linestyles = ["-.", "-", "--", ":"]
    ax[k,0].plot(ps, p2, label=r"$n=10^2$",linestyle=linestyles[0],color='black',)
    ax[k,0].plot(ps, p3, label=r"$n=10^3$",linestyle=linestyles[1],color='black',)
    ax[k,0].plot(ps, p4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
    ax[k,0].axvline(x=(1-q)/2, linestyle="dotted", color="red", linewidth=2)
    ax[k,0].set_ylabel(r"Type I + II error")
    ax[k,0].set_xlabel(r"varing $p$ with $q=0.4$ fixed")
    ax[k,0].legend(frameon=False)

    ax[k,1].plot(ps, q2, label=r"$n=10^2$",linestyle=linestyles[0],color='black')
    ax[k,1].plot(ps, q3, label=r"$n=10^3$",linestyle=linestyles[1],color='black')
    ax[k,1].plot(qs, q4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
    ax[k,1].axvline(x=1-2*p, linestyle="dotted", color="red", linewidth=2)
    ax[k,1].set_ylabel(r"Type I + II error")
    ax[k,1].set_xlabel(r"varing $q$ with $p=0.3$ fixed")
    ax[k,1].legend(frameon=False)

    N = 20
    ps = np.linspace(0.01, 1, N)
    qs = np.linspace(lower_q(K, 10000), 1, N)
    if use_mask is True:
        name0 = "8contour/contour-20-100-5-10000-withmask"
    else:
        name0 = "8contour/contour-20-100-5-10000-nomask"
    save_dict =json.load(open( name0+".json", 'r'))
    z = save_dict["Z"]
    xv, yv = np.meshgrid(ps, qs)
    CS = ax[k,2].contourf(xv, yv, z, linewidths = 1,cmap=plt.cm.bone,vmax=1.)

    ax[k,2].clabel(CS, inline=True, fontsize=10)
    x = np.linspace(0, 0.5, 100)
    ax[k,2].plot(x, 1-2*x, linestyle="dotted", color="red", linewidth=2)
    cbar = fig.colorbar(CS)
    ax[k,2].set_xlabel(r"$p$")
    ax[k,2].set_ylabel(r"$q$")

    plt.tight_layout()
    plt.savefig(f'8contour/HC_both.pdf', dpi=300)
