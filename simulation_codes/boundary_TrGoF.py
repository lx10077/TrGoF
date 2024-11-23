#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy.integrate as integrate
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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
from scipy.stats import gamma, norm

K = 5
c = 5
N_trial = 1000
generate_data = True
acc_eps=1e-10
mask=True


def lower_q(K, n):
    low = np.log(K/(K-1))/np.log(n)
    print(low)
    return low


def compute_score(Ys, alpha=1., s=2,eps=acc_eps):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps)
    n = len(ps)
    first = int(len(ps) * alpha)
    ps = ps[:first]
    rk = np.arange(1,1+first)/n 

    if mask:
        # if s == 0:
        #     ind = (ps >= 1/n*) * (rk >= ps)
        # else:
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

def dominate_Ps(Delta):
    Ps = np.ones(K)
    Tail_Ps = np.ones(K-1)/(K-1)
    Ps[0] = 1-Delta
    Ps[1:] = Delta * Tail_Ps
    A = Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 
    if not A:
        embed()
    return Ps


def modify_date(raw_data, c, modify_n, Delta):
    n = len(raw_data)
    assert modify_n <= n-c, f'{modify_n},{n},{c}'
    modify_set = np.random.randint(c, n, size=modify_n)
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


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))


def experiment_on_p(q, n, s):
    print("=====> With q=",q , "n=", n, 5*np.log(np.log(n)))
    quantiles = np.linspace(0, 6*np.log(np.log(n)), 1000)

    # quantile = np.sqrt((2+0.2)*np.log(np.log(n)))
    # print(quantile)
    estimate = []

    for p in ps:
        eps = n**(-p)
        Delta = n**(-q)

        score_H0 = []
        score_H1 = []

        for _ in tqdm(range(N_trial)):
            raw_data = np.random.uniform(size=n)
            H0s = compute_score(raw_data, s=s)
            score_H0.append(H0s)

            mofidy_data = modify_date(raw_data, c, int(eps*n), Delta)
            H1s = compute_score(mofidy_data, s=s)
            score_H1.append(H1s)

        error = 2
        for quantile in quantiles:
            rj0 = np.sum(score_H0 >= quantile)/len(score_H0)
            rj1 = np.sum(score_H1 <= quantile)/len(score_H1)
            if rj0+rj1 <= error:
                error = rj0+rj1
        print(p, rj0, rj1, error)
        estimate.append(error)
    print(estimate)
    return estimate


def experiment_on_q(p, n, s):
    print("=====> With p=",p , "n=", n)
    quantiles = np.linspace(-30, 30, 1000)

    estimate = []
    lower_bound = lower_q(K,n)

    for q in qs:
        if q <= lower_bound:
            estimate.append(0)
            continue

        eps = n**(-p)
        Delta = n**(-q)

        score_H0 = []
        score_H1 = []

        for _ in tqdm(range(N_trial)):
            raw_data = np.random.uniform(size=n)
            H0s = compute_score(raw_data, s=s)
            score_H0.append(H0s)

            mofidy_data = modify_date(raw_data, c, int(eps*n), Delta)
            H1s = compute_score(mofidy_data, s=s)
            score_H1.append(H1s)

        score_H0 = np.array(score_H0)
        score_H1 = np.array(score_H1)

        error = 2
        for quantile in quantiles:
            rj0 = np.sum(score_H0 >= quantile)/len(score_H1)
            rj1 = np.sum(score_H1 <= quantile)/len(score_H1)
            if rj0+rj1 <= error:
                error = rj0+rj1
        print(q, rj0, rj1, error)
        estimate.append(error)
    print(estimate)
    return estimate

q = 0.4
p = 0.3
s = 0
print(K, s)

name = f'final-{p}-{q}-{K}-{N_trial}-{s}-p'
ps = np.linspace(0.01, 1, 20)
qs = np.linspace(0.01, 1, 20)
if generate_data:
    p2 = experiment_on_p(q, 100, s)
    p3 = experiment_on_p(q, 1000, s)
    p4 = experiment_on_p(q, 10000, s)

    q2 = experiment_on_q(p, 100, s)
    q3 = experiment_on_q(p, 1000, s)
    q4 = experiment_on_q(p, 10000, s)

    save_dict = dict()
    save_dict["p2"] = p2
    save_dict["p3"] = p3
    save_dict["p4"] = p4

    save_dict["q2"] = q2
    save_dict["q3"] = q3
    save_dict["q4"] = q4

    json.dump( save_dict, open( name+".json", 'w'))

else:
    save_dict =json.load(open( name+".json", 'r'))

    p2 = save_dict["p2"] 
    p3 = save_dict["p3"] 
    p4 = save_dict["p4"] 
    # p5 = save_dict["p5"]

    q2 = save_dict["q2"] 
    q3 = save_dict["q3"] 
    q4 = save_dict["q4"]
    # q5 = save_dict["q5"]


linestyles = ["-.", "-", "--", ":"]
ax[0].plot(ps, p2, label=r"$n=10^2$",linestyle=linestyles[0],color='black',)
ax[0].plot(ps, p3, label=r"$n=10^3$",linestyle=linestyles[1],color='black',)
ax[0].plot(ps, p4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
# ax[0].plot(ps, p5, label=r"$n=800$",linestyle=linestyles[3],color="black")
ax[0].axvline(x=(1-q)/2, linestyle="dotted", color="red")
# ax[0].set_aspect('equal')
# ax[0].set_title(r"Fix $q=0.4$")


ax[0].set_ylabel(r"Type I + II error")
ax[0].set_xlabel(r"varing $p$ with $q=0.4$ fixed")
ax[0].legend()

ax[1].plot(ps, q2, label=r"$n=10^2$",linestyle=linestyles[0],color='black')
ax[1].plot(ps, q3, label=r"$n=10^3$",linestyle=linestyles[1],color='black')
ax[1].plot(qs, q4, label=r"$n=10^4$",linestyle=linestyles[2],color="black")
# ax[1].plot(qs, q5, label=r"$n=800$",linestyle=linestyles[3],color="black")
ax[1].axvline(x=1-2*p, linestyle="dotted", color="red")
# ax[1].set_aspect('equal')
# ax[1].set_title(r"Fix $p=0.3$")

ax[1].set_ylabel(r"Type I + II error")
ax[1].set_xlabel(r"varing $q$ with $p=0.3$ fixed")
ax[1].legend()



plt.tight_layout()
plt.savefig(f'{name}.pdf', dpi=300)

