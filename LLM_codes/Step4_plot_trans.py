#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import gamma, norm
import torch
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
parser.add_argument('--language',default="french",type=str)
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
temp = args.temp
alpha = args.alpha
import pickle



def compute_score(Ys, alpha=1., s=2,eps=1e-10, mask=True):
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
        final = (1-ps**(1-s)/(rk+eps)**(-s)-(1-ps)**(1-s)/(1-rk+eps)**(-s))/(s*(1-s))
    
    if mask:
        ind = (ps >= 1/m)* (rk >= ps)
        final *= ind
        
    return m*np.max(final,axis=-1)


print("Begining ploting!!")

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
    # Ys_sum = np.sum(h_ars_Ys, axis=-1)/np.sqrt(given_m)
    return h_ars_Ys

linestyles = [ ":", "-.", "-", "--", ]
colors = ['#1f77b4','#ff7f0e','#d62728','#2ca02c','#9467bd','#8c564b','#e377c2','#7f7f7f','#17becf']

def Log_score(Ys, ratio):
    m = (Ys.shape)[-1]
    given_m = int(ratio*m)
    truncated_Ys = Ys[...,:given_m]
    h_ars_Ys = np.log(truncated_Ys)
    # Ys_sum = np.sum(h_ars_Ys, axis=-1)/np.sqrt(given_m)
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




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 12,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})



def plot_length_on_axis(current_ax, Y, alpha, x_name, legend=True, H="H1", mask=False):
    used_m = Y.shape[-1]
    x = np.arange(1,1+used_m)
    result_set = defaultdict(dict)
    different_s = ["log", "ars", 2, 1.5, 1, 0.5, 0, "opt-0.3", "opt-0.2", "opt-0.1"]
    j = -1
    start_point=3
    for s in different_s:
        j += 1
        if type(s) is not str:
            # x1 = np.arange(200,2+used_m, 10 ).tolist()
            # x0 = np.arange(1,200).tolist()
            # x = np.array(x0+x1)
            x = np.arange(1,1+used_m)
            y = []
            for x_point in x:
                HC, log_critical_value = HC_for_a_given_fraction(Y, x_point, alpha, s, mask=mask)
                mean = np.mean(np.log(HC+1e-10) >= log_critical_value)
                y.append(mean)
            y = np.array(y)[start_point:]
            x = x[start_point:]
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
    current_ax.set_xlabel(rf"{x_name}")
    return result_set


all_tempretures = args.all_temp
this = args.language


for num in [args.non_wm_temp]:
    latter = f"nsiuwm-{num}"
    for mask in [True, False]:
        fig, ax = plt.subplots(nrows=1, ncols=len(all_tempretures), figsize=(4*len(all_tempretures),4))
        final_result = dict()

        for j, temp in enumerate(all_tempretures):
            exp_name1 = f"result/{size}B-robust-c{c}-m{m}-T{T}-{args.seed_way}-{key}-temp{temp}-{this}-trans-{latter}.pkl"
            results1 = pickle.load(open(exp_name1, "rb"))
            sub_Ys = np.array(results1["trans"])[0]
            print("Shape Y:", sub_Ys.shape)

            result = plot_length_on_axis(ax[j], sub_Ys[:,-200:], alpha,f"temp={temp}", legend=False, mask=mask)
            final_result[temp] = result

        name = "trans" ## previous

        save_dir = f"trans_result/{size}B-{name}-c{c}-m{m}-T{T}-tempall-alpha{alpha}-{this}-{mask}-{latter}.pkl"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        pickle.dump(final_result, open(save_dir, "wb"))

        plt.legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
        plt.tight_layout()
        fig_dir = f"trans_fig/{size}B-{name}-c{c}-m{m}-T{T}-translation-tempall-alpha{alpha}-{this}-{mask}-{latter}.pdf"
        os.makedirs(os.path.dirname(fig_dir), exist_ok=True)
        plt.savefig(fig_dir, dpi=300)
