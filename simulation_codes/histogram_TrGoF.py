#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from IPython import embed
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

K = 1000
c = 5
n_e = 3
p = 0.2
q = 0.5
mask = True
direction = None
save_data = True

# n = 1e6
# eps = 1e-3
n = int(10**n_e)
eps = n**(-p)
print("eps", eps)
print("replace", int(n*eps))
Delta = n**(-q)
# Delta = 0.85
print("Delta", Delta)
print(n*eps**2*Delta)


N_trial = 1000
acc_eps=1e-10
name = f'uniform-robust-{N_trial}({K}-n{n_e}-p{p}-q{q}{direction})-{mask}'
s_lst = [2,1.5,1,0.5, 0]



def Zipf(a=1., b=0.01, support_size=5):
    support_Ps = np.arange(1, 1+support_size)
    support_Ps = (support_Ps + b)**(-a)
    support_Ps /= support_Ps.sum()
    return support_Ps


def compute_score(Ys, alpha=1., s=2,eps=acc_eps):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps)
    n = len(ps)
    first = int(len(ps) * alpha)
    ps = ps[:first]
    rk = np.arange(1,1+first)/n 
    # ind = (ps >= 1/n) * (rk >= ps)

    if mask:
        ind = (ps >= 1/n) * (rk >= ps)
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
        
    return n*np.max(final)


# def dominate_Ps(Delta):
#     a = np.random.uniform(0.95, 1.5)
#     b = np.random.uniform(0.01,0.1)
#     support_size = np.random.randint(low=5, high=15)

#     Head_Ps = Zipf(a=a, b=b, support_size=support_size)
#     b = (1 - Delta)/Head_Ps.max()
#     Ps = np.ones(K)
#     if b <= 1:
#         Tail_Ps = np.ones(K-support_size)/(K-support_size)
#         Ps[:support_size] = b * Head_Ps
#         Ps[support_size:] = (1-b) * Tail_Ps
#     else:
#         Ps[0] = 1-Delta
#         Ps[1:1+support_size] = Head_Ps * Delta /2
#         Tail_Ps = np.ones(K-support_size-1)/(K-support_size-1)
#         Ps[1+support_size:] = Delta /2 * Tail_Ps
#     assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3
#     return Ps


def dominate_Ps(Delta):
    if 1-Delta <= 1/K:
        return np.ones(K)/K
    a = np.random.uniform(0.95, 1.5)
    b = np.random.uniform(0.01,0.1)
    Head_Ps = Zipf(a=a, b=b, support_size=K-1)
    b = (1 - Delta)/Head_Ps.max()
    Ps = np.ones(K)
    Ps[0] = max(1-Delta, 1/K)
    Ps[1:] = Head_Ps * Delta
    assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3
    return Ps


def uniform_Ps(Delta):
    if 1-Delta <= 1/K:
        return np.ones(K)/K
    Ps = np.ones(K)/(K-1)*Delta
    Ps[0] = 1-Delta
    assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3
    return Ps


def hard_Ps(Delta):
    Ps = np.zeros(K)
    Ps[0] = 1-Delta
    Ps[1] = Delta
    # assert Ps.max() <= 1-Delta+1e-5 and Ps.max() >= 1-Delta-1e-5 and np.abs(np.sum(Ps)- 1)<= 1e-3
    return Ps

import copy
def modify_date(raw_data, c, modify_n, Delta, choose_type="dom"):
    n = len(raw_data)
    raw_data = copy.deepcopy(raw_data)
    assert modify_n <= n-c
    modify_set = np.random.randint(c, n, size=modify_n)
    for t in modify_set:
        if direction is None:
            Delta_ins = Delta
        if direction == "+":
            Delta_ins = np.random.uniform(Delta, 1)
        else:
            Delta_ins = np.random.uniform(0, Delta)

        if choose_type=="dom":
            Probs = dominate_Ps(Delta_ins)
        elif choose_type=="uni":
            Probs = uniform_Ps(Delta_ins)
        else:
            raise ValueError
        
        # random.seed(tuple(raw_data[(t-c-1):t]+[key]))
        uniform_xi = np.random.uniform(size=K)
        next_token = np.argmax(uniform_xi ** (1/Probs))
        raw_data[t] = uniform_xi[next_token]
    return raw_data


## Generating samples

def get_sample_from(s):
    score_H0 = []
    score_H1 = []
    score_H2 = []

    for _ in tqdm(range(N_trial)):
        raw_data = np.random.uniform(size=n)
        H0s = compute_score(raw_data,s=s)
        score_H0.append(H0s)

        # print("H_0", np.array(score_H0).mean())
        mofidy_data1 = modify_date(raw_data, c, int(eps*n), Delta, "dom")
        # print("H_1", mofidy_data.mean())
        H1s = compute_score(mofidy_data1,s=s)
        score_H1.append(H1s)

        raw_data = np.random.uniform(size=n)
        # print("H_0", np.array(score_H0).mean())
        mofidy_data2 = modify_date(raw_data, c, int(eps*n), Delta, "uni")
        # print("H_1", mofidy_data.mean())
        H2s = compute_score(mofidy_data2,s=s)
        score_H2.append(H2s)
    return np.array(score_H0), np.array(score_H1), np.array(score_H2)


# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

color1 = 'tab:gray'
color2 = "black"
color3 = "white"
alpha = 0.05
Nbins = 30
use_density = True

# s_lst = [2,1, 0.5, 0, -0.5,-1]
# s_lst = [2, 1.75, 1.5,1.25, 1,0.5, 0.25,0]

fig, ax = plt.subplots(ncols=len(s_lst), nrows=3, figsize=(len(s_lst)*3, 9))
print(ax.shape)

results_dict=dict()
ax[0,0].set_ylabel(r"Density")
ax[1,0].set_ylabel(r"Density")
ax[2,0].set_ylabel(r"Density")


if save_data:
    for i, s in enumerate(s_lst):
        print("Used parameter s:", s)
        score_H0, score_H1, score_H2 = get_sample_from(s)
        results_dict[str(s)+"-"+str(s)+"-H0"] = score_H0.tolist()
        results_dict[str(s)+"-"+str(s)+"-H1"] = score_H1.tolist()
        results_dict[str(s)+"-"+str(s)+"-H2"] = score_H2.tolist()

    json.dump( results_dict, open( name+".json", 'w'))
else:
    results_dict = json.load(open( name+".json", 'r'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)


def row(i):
    return 0 

from scipy.stats import gumbel_r

n_column = len(s_lst)
for i, s in enumerate(s_lst): 
    print(i, (row(i), i%n_column))
    score_H0, score_H1, score_H2 = results_dict[str(s)+"-"+str(s)+"-H0"], results_dict[str(s)+"-"+str(s)+"-H1"], results_dict[str(s)+"-"+str(s)+"-H2"]
    score_H0 = np.log(score_H0)
    score_H1 = np.log(score_H1)
    score_H2 = np.log(score_H2)

    _, bins0, patches0 = ax[row(i)][i%n_column].hist(score_H0, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)
    _, bins1, patches1 = ax[row(i)+1][i%n_column].hist(score_H1, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)
    _, bins2, patches2 = ax[row(i)+2][i%n_column].hist(score_H2, bins=Nbins, density=use_density, facecolor = color1, edgecolor=color3, linewidth=0.5)

    quantile = np.quantile(score_H0, 1-alpha)

    for j in range(Nbins):
        if bins0[j] >= quantile:
            patches0[j].set_fc(color2)

    power = np.mean(score_H1>=quantile)
    print(f"With s = {s}, power =", power)
    for j in range(Nbins):
        if bins1[j] >= quantile:
            patches1[j].set_fc(color2)

    power2 = np.mean(score_H2>=quantile)
    print(f"With s = {s}, power =", power2)
    for j in range(Nbins):
        if bins2[j] >= quantile:
            patches2[j].set_fc(color2)

    ax[row(i)][i%n_column].set_title(f"$H_0, s={s}$")
    ax[row(i)][i%n_column].text(0.05, 0.95, rf'$\alpha=0.05$', transform=ax[row(i)][i%n_column].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    ax[row(i)+1][i%n_column].set_title(r"$H_1^{\mathrm{m}},$" + f" $s={s}$")
    ax[row(i)+1][i%n_column].text(0.05, 0.95, rf'$1-\beta={round(power,3)}$', transform=ax[row(i)+1][i%n_column].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    ax[row(i)+2][i%n_column].set_title(r"$H_1^{\mathrm{m}},$" + f" $s={s}$")
    ax[row(i)+2][i%n_column].text(0.05, 0.95, rf'$1-\beta={round(power2,3)}$', transform=ax[row(i)+2][i%n_column].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)
    # ax[row(i)][i%4].set_xlim(xmax=100)


plt.tight_layout()
plt.savefig(name+'-hist.pdf', dpi=300)
