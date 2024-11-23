#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle


different_s = [1, 1.5, 2, "log", "ars", "opt-0.1"]
colors = ['red',"purple" ,'green','gray','blue',"black"]
linestyles = ["-", "-.", "--", ":", "-.", "--"]


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def from_s2label(s):
    if s == "log":
        return r"$h_{\mathrm{log}}$"
    elif s == "ars":
        return r"$h_{\mathrm{ars}}$"
    elif s == "opt-0.1":            
        return r"$h_{\mathrm{opt}, 0.1}$"
    elif s == "opt-0.2":            
        return r"$h_{\mathrm{opt}, 0.2}$"
    elif s == "opt-0.3":            
        return r"$h_{\mathrm{opt}, 0.3}$"
    else:
        return rf"$s={s}$"
    

all_tempretures = [0.1, 0.3, 0.5, 0.7]

alpha = 0.01
mask = True
T = 1000
m = 400
c = 5
num = 0.7

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
star=0
corrupt = 20

for size in ["1p3", "2p7"]:
    fig, ax = plt.subplots(nrows=1, ncols=len(all_tempretures), figsize=(3*len(all_tempretures), 4))
    for i, temp in enumerate(all_tempretures):
        save_dir = f"{size}B-LLMNoLarge-c5-m400-T1000-alpha0.01-temp{temp}-{mask}-nsiuwm-{num}.pkl"
        results_dict = pickle.load(open(save_dir, "rb"))

        result_data = results_dict[corrupt]

        current_ax = ax[i]
        j = -1

        for s in different_s:
            j += 1
            x, y = result_data[s]["x"][star:], result_data[s]["y"][star:]
            current_ax.plot(x,1-np.array(y), label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        if i ==0:
            current_ax.set_ylabel(r"Type II error")
        current_ax.set_xlabel(r"Length of editted text")


        current_ax.text(0.05, 0.95, r'$\mathrm{temp}$ ' + rf'$ = {temp}$', transform=current_ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        if temp >= 0.5:
            current_ax.set_yscale("log")


    # Get handles and labels from one of the subplots for the shared legend
    handles, labels = ax[0].get_legend_handles_labels()

    # Create a shared legend on top of the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(different_s))

    # Adjust layout to make space for the shared legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # ax[-1].legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
    # plt.tight_layout()
    plt.savefig(f"{size}B-adv-c{c}-m{m}-T{T}-tempall-{mask}-all.pdf", dpi=300)
