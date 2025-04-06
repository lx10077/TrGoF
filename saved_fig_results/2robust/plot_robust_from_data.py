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

seed_key = 15485863
segment = 11
c = 5
T = 1000
m = 400


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
    

all_tempretures = [0.1, 0.3, 0.7, 1]
axis_index = [(0, 0), (0, 1), (1, 0), (1, 1)]
axis_index = [0, 1, 2, 3, 4]
nrows, ncols = 3, len(all_tempretures)
log = True

alpha = 0.05
final_result = dict()
mask = True
# props = dict(boxstyle='round', facecolor='white', alpha=0.5)

different_s = [1, 1.5, 2, "log", "ars", "opt-0.1"]
colors = ['red',"purple" ,'green','gray','blue',"black"]
linestyles = ["-", "-.", "--", ":", "-.", "--"]
alpha = 0.01
num=0.7

for size in ["1p3", "2p7"]:
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3+1))
    print(ax)
    for l, task in enumerate(["sub", "ist", "dlt"]):
        if task == "sub":
            axis_name = "Substitution"
        elif task == "ist":
            axis_name = "Insertion"
        else:
            axis_name = "Deletion"

        name = "final"

        save_dir = f"2robust/{size}B-{name}-c5-m400-T1000-tempall-alpha{alpha}-True-{task}-nsiuwm-{num}-11.pkl"
        results_dict = pickle.load(open(save_dir, "rb"))
        save_dir1 = f"2robust/{size}B-{name}-c5-m400-T1000-tempall-alpha{alpha}-False-{task}-nsiuwm-{num}-11.pkl"
        results_dict1 = pickle.load(open(save_dir1, "rb"))
        save_dir2 = f"2robust/{size}B-{name}-c5-m400-T1000-tempall-alpha{alpha}-True1e-3-{task}-nsiuwm-{num}-11.pkl"
        results_dict2 = pickle.load(open(save_dir2, "rb"))

        for i, temp in enumerate(all_tempretures):
            result_data = results_dict[temp]
            result_data1 = results_dict1[temp]
            result_data2 = results_dict2[temp]

            if temp == 0.7:
                a = np.mean(np.minimum(np.minimum(result_data[s]["y"],result_data1[s]["y"]),result_data2[s]["y"]) -np.array(result_data["ars"]["y"]))
                print(temp, task, a)
            

            current_ax = ax[l][i]
            j = -1
            for s in different_s:
                j += 1
                x, y = result_data[s]["x"], np.minimum(np.minimum(result_data[s]["y"],result_data1[s]["y"]),result_data2[s]["y"])
                if temp <= 0.3:
                    x, y = x[:5], y[:5]
                if temp == 0.5:
                    x, y = x[2:8], y[2:8]
                elif temp == 0.7:
                    x, y = x[2:8], y[2:8]
                elif temp == 1:
                    x, y = x[4:10], y[4:10]

                current_ax.plot(x,y, label=from_s2label(s), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            if log == "True":
                current_ax.set_yscale('log')
            if i == 0:
                current_ax.set_ylabel(r"Type II error")
            current_ax.set_xlabel(f"{axis_name} fraction")

            current_ax.text(0.05, 0.95, r'$\mathrm{temp}$ ' + rf'$ = {temp}$', transform=current_ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='left')
            plt.savefig(f"2robust/{size}B-corruption-c{c}-m{m}-T{T}-tempall-mix-all.pdf", dpi=300)

    # Get handles and labels from one of the subplots for the shared legend
    handles, labels = ax[0, 0].get_legend_handles_labels()

    # Create a shared legend on top of the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=len(different_s), frameon=False)

    # Adjust layout to make space for the shared legend
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ax[-1].legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
    # plt.tight_layout()
    plt.savefig(f"2robust/{size}B-corruption-c{c}-m{m}-T{T}-tempall-mix-all.pdf", dpi=300)
