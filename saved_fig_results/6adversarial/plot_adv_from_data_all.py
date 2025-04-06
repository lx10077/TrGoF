#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
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


different_s = [1, 1.5, 2, "log", "ars", "opt-0.1"]
colors = ['red',"purple" ,'green','gray','blue',"black"]
linestyles = ["-", "-.", "--", ":", "-.", "--"]


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
axis_index = [0, 1, 2, 3]
alpha = 0.01
mask = True
T = 1000
num = 0.7
star = 0
c = 5
m = 400
for size in ["1p3", "2p7"]:
    alpha = 0.01
    final_result = dict()

    all_corrupts = [10, 20, 40]
    fig, ax = plt.subplots(nrows=len(all_corrupts), ncols=len(all_tempretures), figsize=(3*len(all_tempretures)+3, 3*len(all_corrupts)+3))

    for l, corrupt in enumerate(all_corrupts):
        for i, temp in enumerate(all_tempretures):
            save_dir = f"6adversarial/{size}B-LLMNoLarge-c5-m400-T1000-alpha0.01-temp{temp}-{mask}-nsiuwm-{num}.pkl"
            results_dict = pickle.load(open(save_dir, "rb"))

            result_data = results_dict[corrupt]
            current_ax = ax[l][i]
            j = -1

            for s in different_s:
                j += 1
                x, y = result_data[s]["x"][star:], result_data[s]["y"][star:]
                current_ax.plot(x,1-np.array(y), label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            if i ==0:
                current_ax.set_ylabel(r"Type II error")
            current_ax.set_xlabel(r"Text length")

            frac = int(corrupt/4)
            text = r'$\mathrm{temp}$ ' + rf'$ = {temp}$, ' + r'$\mathrm{corr}$ ' + rf'$ = {frac}$\%'
            current_ax.set_title(text)

            if (temp >= 0.5 and corrupt <= 20) or (temp == 0.7 and corrupt == 40):
                current_ax.set_yscale("log")

    # Get handles and labels from one of the subplots for the shared legend
    handles, labels = ax[0][0].get_legend_handles_labels()

    # Create a shared legend on top of the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(different_s), frameon=False)

    # Adjust layout to make space for the shared legend
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig(f"6adversarial/{size}B-adv-c{c}-m{m}-T{T}-tempall-corrupall-{mask}-all.pdf", dpi=300)
