#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
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

size = "1p3"
c=5
m=400
T=1000
alpha=0.01
mask =True
import pickle

results = dict()
temperatures = [0.1, 0.3, 0.5, 0.7]

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


def plot_from_data(current_ax, result_set, list_s, x_name, legend=True, H="H1", log=False, truncated_x=None, label=True):
    j=-1
        
    for s in list_s:
        j+=1
        x=np.array(result_set[s]["x"])
        y=np.array(result_set[s]["y"])
        if s in ["log", "ars", "opt-0.1"]:
            smaller_H0 = True
        else:
            smaller_H0 = False

        def intro_gap(x, y, truncated=None):
            used_x = (x <= truncated)
            return x[used_x], y[used_x]
        
        def sparse(x,y, gap=4):
            used_x = np.arange(0, len(x), gap)
            return x[used_x], y[used_x]


        if truncated_x is not None:
            x, y = intro_gap(x, y,truncated=truncated_x)

        if H == "H1":
            current_ax.plot(x, 1-y, label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        else:
            if smaller_H0:
                x, y = sparse(x,y, gap=4)
            else:
                x, y = x[2:], y[2:]
            current_ax.plot(x, y, label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])

    if legend:
        current_ax.legend()
    if H == "H0":
        current_ax.axhline(y=alpha, linestyle=":", color="black")
        if label:
            current_ax.set_ylabel(r"Type I error")
    else:
        if label:
            current_ax.set_ylabel(r"Type II error")
    if log:
        current_ax.set_yscale("log")
    current_ax.set_xlabel(rf"{x_name}")
    pass


def plot_roc_from_data(current_ax, result_set, list_s, legend=True, log=False, filter=1e-3, label=True):
    j=-1
        
    for s in list_s:
        j+=1
        x=np.array(result_set[s]["x"])
        y=np.array(result_set[s]["y"])
        if filter is not None:
            OK = (x >= filter) * (x <= 0.1)
            print(np.sum(OK))
            x = x[OK]
            y = y[OK]
        current_ax.plot(x, y, label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
    if legend:
        current_ax.legend()
    current_ax.set_xlabel(r"Type I error")

    if label:
        current_ax.set_ylabel(r"Type II error")

    if log:
        current_ax.set_yscale("log")
        current_ax.set_xscale("log")
    pass

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
num = 0.7

## For Type II error

for size in ["1p3", "2p7"]:
    nrows, ncols = 2, 4
    fig, axsss = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols+2, 3*nrows+1))
    alpha = 0.01

    if size == "1p3":
        used_roc_length = {0.1:400, 0.3:400, 0.5: 200, 0.7:100, 1:50}  # for 1p3, replace 80 with 100 for 2p7
    else:
        used_roc_length = {0.1:400, 0.3:400, 0.5: 200, 0.7:80, 1:50}  # for 1p3, replace 80 with 100 for 2p7


    for l, temp in enumerate(temperatures):
        print("Working on temperature:", temp)
        cor_level = 0

        save_dir = f"1power/{size}B-powertrun-c{c}-m{m}-T{T}-alpha{alpha}-temp{temp}-{mask}-nsiuwm-{num}.pkl"
        final_result = pickle.load(open(save_dir, "rb"))
        print(final_result.keys())
        H1set = final_result["H0-human"]
        H2set = final_result["H1-watermark"]
        roc_result = final_result["ROC"]

        if temp >= 0.7:
            truncated_x = 150
            use_log = True
        if temp == 1:
            truncated_x = 100
            use_log = True
        if temp == 0.5:
            truncated_x = 200
            use_log = True
        if temp < 0.5:
            truncated_x = None
            use_log = False
        if l == 0:
            y_label = True
        else:
            y_label = False
        plot_from_data(axsss[0,l], H2set, different_s, "Text length", legend=False, H="H1", truncated_x=truncated_x, log=use_log, label=y_label)
        axsss[0,l].text(0.95, 0.95, r'$\mathrm{temp}$ ' + rf'$ = {temp}$', transform=axsss[0,l].transAxes, fontsize=16,
        verticalalignment='top', horizontalalignment='right')

        use_log = True if temp >= 0.3 else False
        plot_roc_from_data(axsss[1,l], roc_result, different_s, legend=False, log=use_log, label=y_label)

        axsss[1,l].text(0.95, 0.95, rf'$n={used_roc_length[temp]}$', transform=axsss[1,l].transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='right')

    # Get handles and labels from one of the subplots for the shared legend
    handles, labels = axsss[0, 0].get_legend_handles_labels()

    # Create a shared legend on top of the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(different_s), fontsize=16, frameon=False)

    # Adjust layout to make space for the shared legend
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"1power/{size}B-robust-c{c}-m{m}-T{T}-tempall-{mask}-all-tight{num}.pdf", dpi=300)

## For Type I error

num = 0.7
size = "1p3"
fig, axsss = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
alpha = 0.01
save_dir = f"1power/{size}B-powertrun-c{c}-m{m}-T{T}-alpha{alpha}-temp0.7-{mask}-nsiuwm-{num}.pkl"
final_result = pickle.load(open(save_dir, "rb"))
print(final_result.keys())
H1set = final_result["H0-human"]
H2set = final_result["H1-watermark"]
roc_result = final_result["ROC"]

axsss.set_ylim(0.006, 0.014)
plot_from_data(axsss, H1set, different_s, "Text length", legend=False, H="H0", truncated_x=None, label=True)
# Get handles and labels from one of the subplots for the shared legend
handles, labels = axsss.get_legend_handles_labels()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
plt.tight_layout()
plt.savefig(f"1power/{size}B-power-null.pdf", dpi=300)