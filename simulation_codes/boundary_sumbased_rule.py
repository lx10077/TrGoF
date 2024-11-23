#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math as ma
import numpy as np
from tqdm import tqdm
from pprint import pprint
from IPython import embed
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
plt.rcParams["font.family"] = "Times New Roman"
from scipy.stats import gamma, norm

K = 1000

generate_data = True

def Zipf(a=1., b=0.01, support_size=5):
    support_Ps = np.arange(1, 1+support_size)
    support_Ps = (support_Ps + b)**(-a)
    support_Ps /= support_Ps.sum()
    return support_Ps


def dominate_Ps(Delta):
    Ps = np.ones(K)
    Tail_Ps = np.ones(K-1)/(K-1)
    Ps[0] = 1-Delta
    Ps[1:] = Delta * Tail_Ps
    assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3, f'{Ps}'
    return Ps


def modify_date(raw_data, modify_n, Delta):
    n = len(raw_data)
    modify_set = np.random.randint(0, n, size=modify_n)
    for t in modify_set:
        
        Probs = dominate_Ps(Delta)
        uniform_xi = np.random.uniform(size=K)
        next_token = np.argmax(uniform_xi ** (1/Probs))
        raw_data[t] = uniform_xi[next_token]
    return raw_data


# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def lower_q(K, n):
    low = np.log(K/(K-1))/np.log(n)
    print(low)
    return low

# Figure 1

import scipy.integrate as integrate


def experiment(n, N, N_trial, h):

    ps = np.linspace(0.01, 1, N)
    qs = np.linspace(lower_q(K, n), 1, N)

    # quantile = compute_quantile(n,alpha=0.05)
    # ps = np.linspace(0.01, 1, N)

    xv, yv = np.meshgrid(ps, qs)
    z = np.zeros_like(xv)
    print(z.shape, yv.shape)
    L_y, L_x = xv.shape

    Eh0 = integrate.quad(lambda x: h(x), 0, 1)[0]

    for j in range(L_y):
        for i in range(L_x):
            p = xv[j][i]
            q = yv[j][i]

            # if np.abs(q + 2*p - 1) < 0.2:
            #     n = 100000
            # else:
            #     n = 10000
            eps = n**(-p)
            Delta = n**(-q)

            # if 2*p + q >= 1:
            #     Delta *= 0.1

            score_H0 = []
            score_H1 = []

            for _ in tqdm(range(N_trial)):
                raw_data = np.random.uniform(size=n)
                h_Y_0 = (h(raw_data).sum() - Eh0)/np.sqrt(n)/np.log(n)
                score_H0.append(h_Y_0)
                # raw_data = np.random.uniform(size=n)
                mofidy_data = modify_date(raw_data, int(eps*n), Delta)
                h_Y_1 = (h(mofidy_data).sum() - Eh0)/np.sqrt(n)/np.log(n)

                score_H1.append(h_Y_1)

            score_H0 = np.array(score_H0)
            score_H1 = np.array(score_H1)


            s = 2
            for quantile in quantiles:
                rj0 = np.sum(score_H0 >= quantile)/len(score_H0)
                rj1 = np.sum(score_H1 <= quantile)/len(score_H1)
                if rj0+rj1 <= s:
                    s = rj0+rj1
            z[j][i] = s
    print(z)
    return z

fig, ax = plt.subplots(figsize=(4,3))

ax=[ax]
K=5
N = 20
n_e = 4
n = 10**n_e
N_trial = 100
h_name = "opt01"

if h_name == "log":
    h = lambda x: np.log(x)
    quantiles = np.linspace(-20, 0, 1000)

elif h_name == "ars":
    h = lambda x: -np.log(1-x)
    quantiles = np.linspace(8, 60, 1000)

elif h_name == "ind08":
    h = lambda x: x >= delta
    delta = 0.8
    quantiles = np.linspace(-10, 10, 1000)
elif h_name == "opt01":
    h = lambda x: np.log(x**9+x**(1/9))
    quantiles = np.linspace(-10, 10, 1000)

elif h_name == "ind05":
    h = lambda x: x >= delta
    delta = 0.5
    quantiles = np.linspace(-10, 10, 1000)
else:
    raise ValueError

name = f"contour-{N}-{N_trial}-{K}-{n}-{h_name}"
print(name)
# N = 10
# N_trial = 10

ps = np.linspace(0.01, 1, N)
qs = np.linspace(lower_q(K, n), 1, N)

xv, yv = np.meshgrid(ps, qs)
if generate_data:
    z = experiment(n, N, N_trial, h)
    save_dict = dict()
    save_dict["N"]=N
    save_dict["N_trial"]=N_trial
    save_dict["Z"] = np.array(z).tolist()
    json.dump(save_dict, open( name+".json", 'w'))
else:
    save_dict =json.load(open( name+".json", 'r'))
    z = save_dict["Z"]

z = np.array(z) 
print(xv.shape, z.shape)

CS = ax[0].contourf(ps, qs, z, linewidths = 1,cmap=plt.cm.bone,vmax=1)
# CS2 = ax[0].contourf(xv, yv, z, linewidths = 1)

ax[0].clabel(CS, inline=True, fontsize=10)

# x = np.linspace(0, 0.5+1/n_e, 100)
# ax[0].plot(x, 1+2/n_e-2*x, linestyle="dotted", color="red")
x = np.linspace(0, 0.5, 100)
ax[0].plot(x, 0.5-x, linestyle="dotted", color="red")
cbar = fig.colorbar(CS)
ax[0].set_xlabel(r"$p$")
ax[0].set_ylabel(r"$q$")
ax[0].set_xlim(xmin=0, xmax=1)
ax[0].set_ylim(ymin=0, ymax=1)


plt.tight_layout()
plt.savefig(name+'.pdf', dpi=300)

