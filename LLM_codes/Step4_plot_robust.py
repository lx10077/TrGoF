#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math as ma
import numpy as np
from tqdm import tqdm
from IPython import embed
from scipy.integrate import quad, dblquad
from scipy.stats import gamma, norm
import torch
import json
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="Gumbel",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
parser.add_argument('--seed',default=1,type=int)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--alpha',default=0.01,type=float)
parser.add_argument('--all_temp', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7], help="A list of temperatures used to generate watermarked texts.")

parser.add_argument('--substitution', action='store_true', help="If set, substitution will be True; otherwise, it defaults to False.")
parser.add_argument('--deletion', action='store_true', help="If set, deletion will be True; otherwise, it defaults to False.")
parser.add_argument('--insertion', action='store_true', help="If set, insertion will be True; otherwise, it defaults to False.")

args = parser.parse_args()


if args.model == "facebook/opt-1.3b":
    size = "1p3"
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    size = "2p7"
else:
    raise ValueError

key = args.seed
torch.manual_seed(key)
c = args.c
T = args.T
m = args.m
alpha = args.alpha
considered_edits = []
if args.substitution:
    considered_edits.append("sub")
if args.deletion:
    considered_edits.append("ist")
if args.insertion:
    considered_edits.append("dlt")
if not considered_edits:
    raise ValueError("You should consider at least one editing method from [substitution, deletion, insertion].")

import pickle


def compute_score(Ys, mask, alpha=1., s=2,eps=1e-10):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps, axis=-1)
    m = ps.shape[-1]
    first = int(m* alpha)
    ps = ps[...,:first]
    rk = np.arange(1,1+first)/first

    # ind = (ps >= 1/n) * (rk >= ps)
    # print(ind.sum())

    if s == 1:
        final = rk * np.log(rk+eps) - rk*np.log(ps+eps) + (1-rk+eps) * np.log(1-rk+eps) - (1-rk) * np.log(1-ps+eps)
    elif s == 0:
        final = ps * np.log(ps+eps) - ps*np.log(rk+eps) + (1-ps+eps) * np.log(1-ps+eps) - (1-ps) * np.log(1-rk+eps)
    elif s == 2:
        final = (rk - ps)**2/(ps*(1-ps)+eps)/2
    elif s == 1/2:
        final = 2*(np.sqrt(rk)-np.sqrt(ps))**2+2*(np.sqrt(1-rk)-np.sqrt(1-ps))**2
    elif s >= 0:
        final = (1-(rk**s)*(ps+eps)**(1-s)-((1-rk)**s)*((1-ps+eps)**(1-s)))/(s*(1-s))
    elif s == -1:
        final = (rk - ps)**2/(rk*(1-rk)+eps)/2
    else: # we must have -1 < s < 0
        final = (1-ps**(1-s)/(rk+eps)**(-s)-(1-ps)**(1-s)/(1-rk+eps)**(-s))/(s*(1-s))

    if mask:
        ind = (ps >= 1e-3)* (rk >= ps)
        # ind = (ps >= 1/m)* (rk >= ps)
        final *= ind
        
    return m*np.max(final,axis=-1)


print("Begining ploting")


def compute_quantile(m, alpha, s, mask):
    # for _ in range(500):
    qs = []
    for _ in range(10):
        raw_data = np.random.uniform(size=(10000, m))
        H0s = compute_score(raw_data, s=s, mask=mask)
        log_H0s = np.log(H0s+1e-10)
        q = np.quantile(log_H0s, 1-alpha)
        qs.append(q)
    return np.mean(qs,axis=0)


def HC_for_a_given_fraction(Ys, ratio, alpha=0.01, s=2, mask=True):
    # embed()
    m = (Ys.shape)[-1]
    if ratio <= 1 and type(ratio)==float:
        given_m = int(ratio*m)
    else:
        given_m = ratio
    truncated_Ys = Ys[...,1:1+given_m]  # Handle the case where Y could be 2-dim or 3-dim tensor.
    HC = compute_score(truncated_Ys, s=s, mask=mask)
    log_critical_value = compute_quantile(given_m,alpha=alpha, s=s, mask=mask)
    return HC, log_critical_value


def Ars_score(Ys, ratio):
    m = (Ys.shape)[-1]
    given_m = int(ratio*m)
    truncated_Ys = Ys[...,:given_m]
    h_ars_Ys = -np.log(1-truncated_Ys)
    return h_ars_Ys

linestyles = [ ":", "-.", "-", "--", ]
colors = ['#1f77b4','#ff7f0e','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#17becf']


def Log_score(Ys, ratio):
    m = (Ys.shape)[-1]
    given_m = int(ratio*m)
    truncated_Ys = Ys[...,:given_m]
    h_ars_Ys = np.log(truncated_Ys)
    return h_ars_Ys


def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)


def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


def f_opt(r, delta):
    inte_here = np.floor(1/(1-delta))
    rest = 1-(1-delta)*inte_here
    return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))


def h_opt_gum(Ys, delta0=0.2, alpha=0.01):
    # Compute critical values
    Ys = np.array(Ys)
    h_ars_Ys = f_opt(Ys, delta0)
    cumsum_Ys = np.sum(h_ars_Ys, axis=1)/np.sqrt(Ys.shape[1])

    def find_q(N=2500):
        Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
        Simu_Y = f_opt(Null_Ys, delta0)
        Simu_Y = np.sum(Simu_Y, axis=1)/np.sqrt(Ys.shape[1])
        h_help_qs = np.quantile(Simu_Y, 1-alpha)
        return h_help_qs
     
    q_lst = []
    for N in [2500] * 10:
        q_lst.append(find_q(N))
    h_help_qs = np.mean(np.array(q_lst))

    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results)


def clean_data(Y, m=None):
    _, used_m =  Y.shape
    cleaned_Y = np.unique(Y)
    if m is None:
        m = used_m
    new_N = len(cleaned_Y)//m
    cleaned_Y = cleaned_Y[:new_N*m]
    return cleaned_Y.reshape(new_N, m)
    
def remove_repeated(Ys, filered_length=None,filter_data=False,first=500,mute=True):
    unique_elements_num = []
    filtered_rows = []
    # Iterate over each row in the array
    for row in Ys:
        # Use numpy's `np.unique` to remove duplicates while preserving order
        _, unique_indices = np.unique(row, return_index=True)
        unique_row = row[np.sort(unique_indices)]
        unique_elements_num.append(len(unique_row))

        if filered_length is not None and len(unique_row) > filered_length:
            filtered_rows.append(unique_row[:filered_length])

    unique_elements_num = np.array(unique_elements_num)
    if filered_length is not None:
        filtered_rows = np.array(filtered_rows)
    
    if not mute:
        print("mean:", np.mean(unique_elements_num))
        print("If we want at 500 samples, there should >=", 500/len(Ys))
        print(">=250:", np.mean(unique_elements_num>=250))
        print(">=200:", np.mean(unique_elements_num>=200))
        print(">=150:", np.mean(unique_elements_num>=150))
        print(">=100:", np.mean(unique_elements_num>=100))
        print(">=50:", np.mean(unique_elements_num>=50))
        print()

    if filered_length is not None:
        return filtered_rows
    
    if filter_data is True:
        # Find the indices of the 100 largest numbers
        largest_indices = np.argpartition(unique_elements_num, -first)[-first:]

        # Sort the indices by the actual values to get them in descending order
        largest_indices = largest_indices[np.argsort(-unique_elements_num[largest_indices])]

        # Get the largest 500 entries
        largest_values = unique_elements_num[largest_indices]

        print(np.array(largest_values)/Ys.shape[1])
        return Ys[largest_indices]
    return Ys


def plot_robust_on_axis(current_ax, Y, alpha, legend=True, x_point=None, mask=True, norepeat=False):
    ## Assume Y is three dimension tensor
    N_corrup, _, used_m = Y.shape
    delta_gap = 0.05
    x =  np.linspace(0,(N_corrup-1)*delta_gap,N_corrup)
    # assert len(x) == N_corrup, f"len x = {len(x)}, N_corrup = {N_corrup}"
    if x_point is not None and x_point < used_m:
        Y = Y[:,:, -x_point:]
        used_m = Y.shape[-1]
        print("Used_m:", used_m)

    result_set = defaultdict(dict)
    different_s = ["log", "ars", 2, 1.5, 1, 0.5, 0, "opt-0.3", "opt-0.2", "opt-0.1"]
    j = -1
    for s in different_s:
        j += 1
        if type(s) is not str:
            y = []
            for corrup_level in range(N_corrup):
                if norepeat:
                    here_Y = remove_repeated(Y[corrup_level], filered_length=250)
                else:
                    here_Y = Y[corrup_level]

                HC, log_critical_value = HC_for_a_given_fraction(here_Y, 1., alpha, s, mask=mask)
                mean = np.mean(np.log(HC+1e-10) >= log_critical_value)
                y.append(mean)
            y = 1-np.array(y)
            current_ax.plot(x,y, label=f"s={s}", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        elif s == "log":
            y = []
            for corrup_level in range(N_corrup):
                if norepeat:
                    here_Y = remove_repeated(Y[corrup_level], filered_length=250)
                else:
                    here_Y = Y[corrup_level]

                Ylog = Log_score(here_Y, 1)      
                sum_Ys = np.sum(Ylog, axis=1)
                h_log_qs = gamma.ppf(q=alpha,a=used_m)
                results = (sum_Ys >= -h_log_qs)
                mean = np.mean(results,axis=0)
                y.append(mean)
            y = 1-np.array(y)
            current_ax.plot(x,y, label=f"log", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        elif s == "ars":
            y = []
            for corrup_level in range(N_corrup):
                if norepeat:
                    here_Y = remove_repeated(Y[corrup_level], filered_length=250)
                else:
                    here_Y = Y[corrup_level]

                Yars = Ars_score(here_Y, 1)      
                sum_Ys = np.sum(Yars, axis=1)
                h_ars_qs = gamma.ppf(q=1-alpha,a=used_m)
                results = (sum_Ys >= h_ars_qs)
                mean = np.mean(results,axis=0)
                y.append(mean)
            y = 1-np.array(y)
            current_ax.plot(x,y, label=f"ars", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        elif "opt" in s:

            Delta = float(s[4:])
            y = []
            for corrup_level in range(N_corrup):
                if norepeat:
                    here_Y = remove_repeated(Y[corrup_level], filered_length=250)
                else:
                    here_Y = Y[corrup_level]

                mean = h_opt_gum(here_Y, Delta, alpha=alpha)
                y.append(mean)
            y = 1-np.array(y)
            current_ax.plot(x,y, label=s, linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        else:
            raise ValueError

        result_set[s]["x"] = np.array(x).tolist()
        result_set[s]["y"] = np.array(y).tolist()

    # current_ax.set_aspect('equal', 'box')
    if legend:
        current_ax.legend()
    current_ax.set_ylabel(r"Type II error")
    return result_set



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 12,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

all_tempretures = args.all_temp
num = args.non_wm_temp
latter = f"nsiuwm-{num}-11"

for mask in [True, False]:
    for i, task in enumerate(considered_edits):

        fig, ax = plt.subplots(nrows=1, ncols=len(all_tempretures), figsize=(4*len(all_tempretures),4))
        final_result = dict()

        if task == "sub":
            lengs = [400, 400, 400, 200, 200]
        elif task == "dlt":
            lengs = [200, 200, 200, 200, 200]
        else:
            lengs = [400, 400, 400, 200, 200]

        for j, temp in enumerate(all_tempretures):

            exp_name1 = f"result/{size}B-robust-c{c}-m{m}-T{T}-{args.seed_way}-{key}-temp{temp}-{task}-{latter}.pkl"
            results1 = pickle.load(open(exp_name1, "rb"))
            sub_Ys = np.array(results1[task])
            result = plot_robust_on_axis(ax[j], sub_Ys, alpha, legend=False, x_point=lengs[j], mask=mask, norepeat=False)
            final_result[temp] = result

        name = "edit" 

        save_dir = f"corrup_result/{size}B-{name}-c{c}-m{m}-T{T}-tempall-alpha{alpha}-{mask}1e-3-{task}-{latter}.pkl"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        pickle.dump(final_result, open(save_dir, "wb"))

        plt.legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
        plt.tight_layout()
        
        fig_dir = f"corrup_fig/{size}B-{name}-c{c}-m{m}-T{T}-corruption-tempall-alpha{alpha}-{mask}-{task}-{latter}.pdf"
        os.makedirs(os.path.dirname(fig_dir), exist_ok=True)
        plt.savefig(fig_dir, dpi=300)

