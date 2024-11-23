#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import json
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

K = 5
acc_eps = 1e-10
generate_data = True
mask=True


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



def compute_score(Ys, alpha=1., s=2,eps=acc_eps):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps)
    n = len(ps)
    first = int(len(ps) * alpha)
    ps = ps[:first]
    rk = np.arange(1,1+first)/n 

    if mask:
        ind = (ps >= 1/n**2) * (rk >= ps)
        ps = ps[ind]
        rk = rk[ind]

    if s == 1:
        final = rk * np.log(rk+eps) - rk*np.log(ps+eps) + (1-rk+eps) * np.log(1-rk+eps) - (1-rk) * np.log(1-ps+eps)
    elif s == 0:
        final = ps * np.log(ps+eps) - ps*np.log(rk+eps) + (1-ps+eps) * np.log(1-ps+eps)- (1-ps) * np.log(1-rk+eps)
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

    return np.log(n*np.max(final))


def experiment(n, N, N_trial, s):
    ps = np.linspace(0.01, 1, N)
    qs = np.linspace(lower_q(K, n), 1, N)

    quantiles = np.linspace(0, 30, 1000)

    xv, yv = np.meshgrid(ps, qs)
    z = np.zeros_like(xv)
    L_y, L_x = xv.shape

    for j in range(L_y):
        for i in range(L_x):
            p = xv[j][i]
            q = yv[j][i]

            eps = n**(-p)
            Delta = n**(-q)


            score_H0 = []
            score_H1 = []

            for _ in tqdm(range(N_trial)):
                raw_data = np.random.uniform(size=n)
                H0s = compute_score(raw_data, s=s)
                score_H0.append(H0s)

                mofidy_data = modify_date(raw_data, int(eps*n), Delta)
                H1s = compute_score(mofidy_data, s=s)
                score_H1.append(H1s)

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
N = 20
n_e = 4
n = 10**n_e
N_trial = 100
s=0

name = f"contour-{N}-{N_trial}-{K}-{n}-{s}"
print(name)
# N = 10
# N_trial = 10

ps = np.linspace(0.01, 1, N)
qs = np.linspace(lower_q(K, n), 1, N)

xv, yv = np.meshgrid(ps, qs)
if generate_data:
    z = experiment(n, N, N_trial, s)
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
ax[0].plot(x, 1-2*x, linestyle="dotted", color="red")
cbar = fig.colorbar(CS)
ax[0].set_xlabel(r"$p$")
ax[0].set_ylabel(r"$q$")
ax[0].set_xlim(xmin=0, xmax=1)
ax[0].set_ylim(ymin=0, ymax=1)

plt.tight_layout()
plt.savefig(name+'.pdf', dpi=300)
