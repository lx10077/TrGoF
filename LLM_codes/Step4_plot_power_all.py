#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math as ma
import numpy as np
from tqdm import tqdm
from IPython import embed
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
parser.add_argument('--temp',default=0.1, type=float)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--batch_size',default=10,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--N',default=4,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language',default="french",type=str)
parser.add_argument('--truncate_vocab',default=8,type=int)
args = parser.parse_args()



key = args.seed
torch.manual_seed(key)
seed_key = 15485863
segment = args.N
c = args.c
T = args.T
m = args.m
mask = True

import pickle


def compute_score(Ys, alpha=1., s=2,eps=1e-10, mask=mask):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps, axis=-1)
    m = ps.shape[-1]
    first = int(m* alpha)
    ps = ps[...,:first]
    rk = np.arange(1,1+first)/first

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
        final = (1-ps**(1-s)/(rk+eps)**(-s)-(1-ps)**(1-s)/(1-rk+eps)**(-s))/(s*(1-s)+eps)
    
    if s>=1 and mask:
        # final[:, :2] = 0
        ind = (ps >= 1/m)* (rk >= ps)
        final *= ind
    elif s < 1:
        final = final
    
    # final = np.minimum(final, 10)

    return m*np.max(final,axis=-1)


print("Begining ploting")


def compute_quantile(m, alpha, s, mask=True):
    # for _ in range(500):
    qs = []
    for _ in range(10):
        raw_data = np.random.uniform(size=(10000, m))
        H0s = compute_score(raw_data, s=s, mask=mask)
        log_H0s = np.log(H0s+1e-10)
        q = np.quantile(log_H0s, 1-alpha)
        qs.append(q)
    return np.mean(qs,axis=0)


def HC_for_a_given_fraction(Ys, ratio, alpha=0.05, s=2, mask=True):
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
    # Ys_sum = np.sum(h_ars_Ys, axis=-1)/np.sqrt(given_m)
    return h_ars_Ys


def Log_score(Ys, ratio):
    m = (Ys.shape)[-1]
    given_m = int(ratio*m)
    truncated_Ys = Ys[...,:given_m]
    h_ars_Ys = np.log(truncated_Ys)
    # Ys_sum = np.sum(h_ars_Ys, axis=-1)/np.sqrt(given_m)
    return h_ars_Ys


def f_opt(r, delta):
    inte_here = np.floor(1/(1-delta))
    rest = 1-(1-delta)*inte_here
    return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    

def h_opt_gum(Ys, delta0=0.1, alpha=0.05):
    # Compute critical values
    Ys = np.array(Ys)
    check_points = np.arange(1, 1+Ys.shape[-1])
    h_ars_Ys = f_opt(Ys, delta0)
    
    def find_q(N=2500):
        Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
        Simu_Y = f_opt(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)
        return h_help_qs
    
    q_lst = []
    for N in [2500] * 10:
        q_lst.append(find_q(N))
    h_help_qs = np.mean(np.array(q_lst),axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results,axis=0)



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


def plot_length_on_axis(current_ax, Y, alpha, x_name, legend=True, H="H1", KGW=None, log=False):
    global different_s
    used_m = Y.shape[-1]
    print(H, Y.shape)

    x = np.arange(1,1+used_m)
    result_set = defaultdict(dict)
    j = -1
    for s in different_s:
        j += 1
        if type(s) is not str:
            x = np.arange(1,1+used_m, int(used_m/100))
            y = []
            for x_point in x:
                HC, log_critical_value = HC_for_a_given_fraction(Y, x_point, alpha, s, mask=mask)
                mean = np.mean(np.log(HC+1e-10) >= log_critical_value)
                y.append(mean)
            y = np.array(y)
            if H == "H1":
                current_ax.plot(x, 1-y, label=f"s={s}", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            else:
                current_ax.plot(x, y, label=f"s={s}", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
                
        elif s == "log":
            Ylog = Log_score(Y, 1)      
            cumsum_Ys = np.cumsum(Ylog, axis=1)
            h_log_qs = compute_gamma_q(alpha, x)
            results = (cumsum_Ys >= -h_log_qs)
            y = np.mean(results,axis=0)
            if H == "H1":
                current_ax.plot(x,1-y, label=f"log", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            else:
                current_ax.plot(x,y, label=f"log", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])

        elif s == "ars":
            Yars = Ars_score(Y, 1)    
            cumsum_Ys = np.cumsum(Yars, axis=1)
            h_ars_qs = compute_gamma_q(1-alpha, x)
            results = (cumsum_Ys >= h_ars_qs)
            y = np.mean(results,axis=0)
            if H == "H1":
                current_ax.plot(x, 1-y, label=f"ars", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            else:
                current_ax.plot(x, y, label=f"ars", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        
        elif "opt" in s:
            Delta = float(s[4:])
            y = h_opt_gum(Y, Delta, alpha=alpha)
            x = np.arange(1,1+used_m)
            if H == "H1":
                current_ax.plot(x, 1-y, label=f"opt-{Delta}", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
            else:
                current_ax.plot(x, y, label=f"opt-{Delta}", linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])

        else:
            raise ValueError
        result_set[s]["x"] = x.tolist()
        result_set[s]["y"] = y.tolist()

    j+=1

    # current_ax.set_aspect('equal', 'box')
    if legend:
        current_ax.legend()
    if H == "H0":
        current_ax.axhline(y=alpha, linestyle="--", color="black")
        current_ax.set_ylabel(r"Type I error")
    else:
        current_ax.set_ylabel(r"Type II error")
    if log:
        current_ax.set_yscale("log")
    current_ax.set_xlabel(rf"{x_name}")
    return result_set


def plot_ROC(current_ax, null_Y, gum_Y, truncated=None):
    global different_s
    if truncated is not None and type(truncated) == int:
        null_Y = null_Y[:, :truncated]
        gum_Y = gum_Y[:, :truncated]
    used_m = gum_Y.shape[-1]

    null_Y = np.array(null_Y)
    gum_Y = np.array(gum_Y)
    result_set = defaultdict(dict)

    def compute_x_y(H0Y, H1Y):
        min_v = min(np.min(H0Y), np.min(H1Y))
        max_v = max(np.max(H0Y), np.max(H1Y))
        if min_v >= 0:
            th_range = np.linspace(0, 1.1*max_v, 5000)
        else:
            th_range = np.linspace(1.1*min_v, 1.1*max_v, 5000)
        x = []
        y = []
        for th in th_range:
            rj0 = (H0Y >= th).mean()
            rj1 = (H1Y <= th).mean()
            x.append(rj0)
            y.append(rj1)
        return np.array(x), np.array(y)
    
    j=-1
    for s in different_s:
        j += 1
        if type(s) is not str:
            H0Y = np.log(compute_score(null_Y, s=s))
            H1Y = np.log(compute_score(gum_Y, s=s))
            print("s", H0Y.shape, H1Y.shape)
            x, y = compute_x_y(H0Y, H1Y)
            label = f"s={s}"
                
        elif s == "log":
            H0Y = Log_score(null_Y, 1)
            H1Y = Log_score(gum_Y, 1)    
            cumsum_H0Y = np.sum(H0Y, axis=1)/np.sqrt(used_m)
            cumsum_H1Y = np.sum(H1Y, axis=1)/np.sqrt(used_m)
            x, y = compute_x_y(cumsum_H0Y, cumsum_H1Y)
            label = "log"

        elif s == "ars":
            H0Y = Ars_score(null_Y, 1)    
            H1Y = Ars_score(gum_Y, 1)    
            cumsum_H0Y = np.sum(H0Y, axis=1)/np.sqrt(used_m)
            cumsum_H1Y = np.sum(H1Y, axis=1)/np.sqrt(used_m)
            x, y = compute_x_y(cumsum_H0Y, cumsum_H1Y)
            label = "ars"
        
        elif "opt" in s:
            Delta = float(s[4:])
            H0Y = f_opt(null_Y, Delta)   
            H1Y = f_opt(gum_Y, Delta)   
            cumsum_H0Y = np.sum(H0Y, axis=1) /np.sqrt(used_m)
            cumsum_H1Y = np.sum(H1Y, axis=1) /np.sqrt(used_m)
            x, y = compute_x_y(cumsum_H0Y, cumsum_H1Y)
            label = s
        
        current_ax.plot(x, y, label=label, linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])
        result_set[s]["x"] = x.tolist()
        result_set[s]["y"] = y.tolist()
    current_ax.plot(np.linspace(0,1,1000), 1-np.linspace(0,1,1000), linestyle="--", color="black")
    current_ax.set_xlabel(r"Type I error")
    current_ax.set_ylabel(r"Type II error")
    
    current_ax.set_yscale("log")
    current_ax.set_xscale("log")
    return result_set
    


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 12,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

linestyles = ["-", "-.", "--","-.", ":", "--", "-.", ":","-.",]
colors = ["tab:blue", "tab:orange", "black", "tab:purple", "tab:red",  "tab:pink", "tab:gray",   "tab:brown", "tab:green", ]
results = dict()
temperatures = [0.1, 0.3, 0.5, 0.7, 1]
# used_roc_length = {0.1:100, 0.3:30, 0.7:10, 1:10}
used_roc_length = {0.1:400, 0.3:400, 0.5:200, 0.7:80, 1:30}


def remove_repeated(Ys, filered_length=None,filter_data=False,first=500):
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
    
    print("mean:", np.mean(unique_elements_num))
    print("If we want at 500 samples, there should >=", 500/len(Ys))
    print(">=250:", np.mean(unique_elements_num>=250))
    print(">=200:", np.mean(unique_elements_num>=200))
    print(">=150:", np.mean(unique_elements_num>=150))
    print(">=100:", np.mean(unique_elements_num>=100))
    print(">=50:", np.mean(unique_elements_num>=50))

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


print("Start plotting~!!")
different_s = ["log", "ars", 2, 1.5, 1, "opt-0.3", "opt-0.2", "opt-0.1", "opt-0.05"]
temperatures = [0.1, 0.3, 0.5, 0.7, 1]
lenght_for_temps = [400, 400, 400, 200, 30]


for num in [args.non_wm_temp]:
    latter = f"nsiuwm-{num}"
    print(latter)
    
    for size in ["1p3"]:
        print(size)
        for k, temp in enumerate(temperatures):
            print()
            print("Working on temperature:", temp)

            Gum_name = f"raw_data/{size}B-Gumbel-c{c}-m{m}-T{T}-{args.seed_way}-15485863-temp{temp}-{latter}.pkl"

            # Gum_name=f"corrupted_note_data/2p7B-Y-c5-m400-T2500-noncomm_prf-1-temp{temp}-sub.pkl"
            Gum_result = pickle.load(open(Gum_name, "rb"))
            Gum_Ys = np.array(Gum_result["watermark"]["Ys"])
            repeated = np.array(Gum_result["watermark"]["where_watermark"])

            print("Load Gum results...\n")
            print(Gum_Ys.shape)
            # Gum_Ys = remove_repeated(Gum_Ys, filter_data=True)

            print("Load Gum null results...\n")
            if "null" not in Gum_result:
                print("copying here")
                here_temp = 1 if (temp == 0.1 or temp == 0.5) else temp
                Gum_name0 = f"/Workspace/Users/lixian@pennmedicine.upenn.edu/robust/result/{size}B-robust-c5-m400-T2500-{args.seed_way}-{key}-temp{here_temp}-sub.pkl"
                null_result = pickle.load(open(Gum_name0, "rb"))

                null_Ys =  np.array(null_result["null"])[0]
                print("Orginal shape:", null_Ys.shape)
                null_Ys = null_Ys[:5000]

                Gum_result["null"] = null_Ys
                pickle.dump(Gum_result, open(Gum_name, "wb"))
            else:
                null_Ys =  np.array(Gum_result["null"])

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,5))
            print("Corruption level is", 0)
            alpha = 0.01
            
            if temp >= 0.5:
                use_log = True
            else:
                use_log = False
            
            # Gum_Ys= remove_repeated(Gum_Ys, filered_length=250)
            # null_Ys = remove_repeated(null_Ys, filered_length=250)
            # print("The Gum_Y shape is:", Gum_Ys.shape)
            # print("The null shape is:", null_Ys.shape)

            H1set = plot_length_on_axis(ax[0], null_Ys, alpha, "Unwatermarked text length", legend=False, H="H0", log=False)
            H2set = plot_length_on_axis(ax[1], Gum_Ys, alpha, "Watermarked text length", legend=True, H="H1", log=use_log)
            ROCset = plot_ROC(ax[2], null_Ys, Gum_Ys, truncated=used_roc_length[temp])

            final_result = dict()
            final_result["H0-human"] = H1set
            final_result["H1-watermark"] = H2set
            final_result["ROC"] = ROCset

            notation = "intro"
            save_dir = f"fig_data/{size}B-{notation}-c{c}-m{m}-T{T}-alpha{alpha}-temp{temp}-{mask}-{latter}.pkl"
            pickle.dump(final_result, open(save_dir, "wb"))

            plt.legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
            plt.tight_layout()
            plt.savefig(f"fig/{size}B-{notation}-c{c}-m{m}-T{T}-alpha{alpha}-temp{temp}-{mask}-{latter}.pdf", dpi=300)
