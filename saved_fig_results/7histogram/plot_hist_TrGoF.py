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

K = 1000
c = 5
n_e = 3
p = 0.2
q = 0.5
N_trial = 1000
direction = None
s_lst = [2, 1.5, 1, 0.5, 0]

color1 = 'tab:gray'
color2 = "black"
color3 = "white"
alpha = 0.05
Nbins = 30
use_density = True
present_M2_result = False

for mask in [True, False]:
    print("\nIf the truncation for Tr-GoF is", mask)
    nrows = 3 if present_M2_result else 2
    fig, ax = plt.subplots(ncols=len(s_lst), nrows=nrows, figsize=(len(s_lst)*3, 3*nrows))

    results_dict=dict()
    ax[0,0].set_ylabel(r"Density")
    ax[1,0].set_ylabel(r"Density")
    if present_M2_result:
        ax[2,0].set_ylabel(r"Density")
    name = f'7histogram/uniform-robust-{N_trial}({K}-n{n_e}-p{p}-q{q}{direction})-{mask}'
    results_dict = json.load(open(name+".json", 'r'))

    n_column = len(s_lst)
    for i, s in enumerate(s_lst): 
        score_H0, score_H1, score_H2 = results_dict[str(s)+"-"+str(s)+"-H0"], results_dict[str(s)+"-"+str(s)+"-H1"], results_dict[str(s)+"-"+str(s)+"-H2"]
        score_H0 = np.log(score_H0)
        score_H1 = np.log(score_H1)
        score_H2 = np.log(score_H2)

        _, bins0, patches0 = ax[0][i%n_column].hist(score_H0, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)
        _, bins1, patches1 = ax[1][i%n_column].hist(score_H1, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)
        quantile = np.quantile(score_H0, 1-alpha)

        if present_M2_result:
            _, bins2, patches2 = ax[2][i%n_column].hist(score_H2, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)
            power2 = np.mean(score_H2>=quantile)
            print(f"With s = {s}, M2, power =", power2)
            for j in range(Nbins):
                if bins2[j] >= quantile:
                    patches2[j].set_fc(color2)

        for j in range(Nbins):
            if bins0[j] >= quantile:
                patches0[j].set_fc(color2)

        power = np.mean(score_H1>=quantile)
        print(f"With s = {s}, M1, power =", power)
        for j in range(Nbins):
            if bins1[j] >= quantile:
                patches1[j].set_fc(color2)

        ax[0][i%n_column].set_title(f"$H_0, s={s}$")
        ax[0][i%n_column].text(0.05, 0.95, rf'$\alpha=0.05$', transform=ax[0][i%n_column].transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left')
        ax[1][i%n_column].set_title(r"$H_1^{\mathrm{mix}},$" + f" $s={s}$")
        ax[1][i%n_column].text(0.05, 0.95, rf'$1-\beta={round(power,3)}$', transform=ax[1][i%n_column].transAxes, fontsize=16,
            verticalalignment='top', horizontalalignment='left')
        if present_M2_result:
            ax[2][i%n_column].set_title(r"$H_1^{\mathrm{mix}},$" + f" $s={s}$")
            ax[2][i%n_column].text(0.05, 0.95, rf'$1-\beta={round(power2,3)}$', transform=ax[2][i%n_column].transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(name+'-hist.pdf', dpi=300)
