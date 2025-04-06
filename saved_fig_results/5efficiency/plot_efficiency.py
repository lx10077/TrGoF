#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman" # Comment this if your computer doesn't support this font
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True, # Comment this if your computer doesn't support latex formula
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
import scipy.special as sc
from scipy.optimize import minimize


N = 1000
Deltas = np.linspace(0., 0.99, N)

# h_ars
def I_h1(dlta, eps = 1):
    def phi_h1(x):
        inte_here = np.floor(1/(1-dlta))
        rest = 1-(1-dlta)*inte_here + 1e-8
        return inte_here*sc.beta(1/(1-dlta), x+1) + sc.beta(1/rest, x+1)
    
    def f(x):
        return x + np.log(eps*phi_h1(x)+(1-eps)/(x+1))
    
    def constraint1(x):
        return x
    
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = [con1]
    sol = minimize(f,0,method='SLSQP',constraints=cons)
    assert sol["x"][0] >= 0
    return -sol["fun"]


# h_log
def I_h2(dlta, eps = 1):
    inte_here = np.floor(1/(1-dlta))
    rest = 1-(1-dlta)*inte_here +1e-8

    def phi_h2(x):
        return inte_here*(1-dlta)/(1-(1-dlta)*x+1e-8) + rest/(1-rest*x+1e-8)
    
    def f(x):
        return -x + np.log(eps*phi_h2(x)+(1-eps)/(1-x)+1e-8) 
    
    def constraint1(x):
        return x
    
    def constraint2(x):
        return 1/(1-dlta) - x
    
    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}

    cons = [con1, con2]
    sol = minimize(f,0,method='SLSQP',constraints=cons)
    assert sol["x"][0] >= 0
    return -sol["fun"]


# h_ind
def compute_h3_R(h3_delta):
    F = (1-Deltas) * h3_delta**(1/(1-Deltas))
    rhs = (1-h3_delta)/(1-F)
    Rh3s = h3_delta * np.log(h3_delta/F) + (1-h3_delta) * np.log(rhs)
    return Rh3s


# hopt, 
def I_opt(dlta, Delta, eps = 1):
    def rho(x, d):
        inte_here = np.floor(1/(1-d))
        rest = 1-(1-d)*inte_here +1e-8
        V = inte_here * x**(d/(1-d)) + x**(1/rest-1)
        return V

    def phi_h1(x):
        return quad(lambda y: rho(y, Delta)**(-x)*rho(y, dlta), 0, 1,epsabs = 1e-8,epsrel=1e-8)[0]
    
    def mu(x):
        return quad(lambda y: rho(y, Delta)**(-x), 0, 1,epsabs = 1e-8,epsrel=1e-8)[0]
    
    mmm = quad(lambda y: np.log(rho(y, Delta)), 0, 1,epsabs = 1e-8,epsrel=1e-8)[0]

    def f(x):
        return mmm*x + np.log(eps*phi_h1(x)+(1-eps)*mu(x))
    
    def constraint1(x):
        return x
    
    def constraint2(x):
        return 1/(1-Delta) - x
    
    con1 = {'type': 'ineq', 'fun': constraint1}
    # con2 = {'type': 'ineq', 'fun': constraint2}
    cons = [con1]
    sol = minimize(f,1e-6,method='SLSQP',constraints=cons)
    assert sol["x"][0] >= 0
    return -sol["fun"]

print()
print("Ars P1:",I_h1(1e-3))
print("Log P1:",I_h2(1e-3))
print()

def compute_h3_R_new(h3_delta, eps = 1):
    inte_here = np.floor(1/(1-Deltas))
    rest = 1-(1-Deltas)*inte_here+1e-8
    F = inte_here * (1-Deltas)*h3_delta**(1/(1-Deltas+1e-8)) + rest * h3_delta **(1/rest)
    F = eps * F + (1-eps) * h3_delta
    Rh3s = h3_delta * np.log(h3_delta/F+1e-8) + (1-h3_delta) * np.log((1-h3_delta)/(1-F)+1e-8)
    return Rh3s


def find_root():
    def f(x):
        return np.abs(I_h1(x) - I_h2(x))
    def constraint1(x):
        return x - 0.1
    
    def constraint2(x):
        return 0.2 - x
    
    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}

    cons = [con1, con2]
    sol = minimize(f,0.15,method='SLSQP',constraints=cons)
    assert sol["x"][0] >= 0
    print(sol["x"][0])

find_root()


def f(r, delta, eps=1):
    inte_here = np.floor(1/(1-delta))
    rest = 1-(1-delta)*inte_here + 1e-8
    V = inte_here*r**(delta/(1-delta))+ r**(1/rest-1)
    V = eps * V + (1-eps)
    return -np.log(V+1e-8)


linestyles = ["--", "-.",":","-", ]
from scipy.integrate import quad

def for_a_delta(delta, eps=1):
    def g(x):
        return f(x, delta, eps)
    return quad(g, 0, 1,epsabs = 1e-6,epsrel=1e-6)[0]
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

eps = 1
Rh_opt = []
for dlta in Deltas:
    Rh_opt.append(for_a_delta(dlta, eps))
Rh_opt = np.array(Rh_opt)

Rh1s = []
for dlta in Deltas:
    Rh1s.append(I_h1(dlta, eps))
Rh1s = np.array(Rh1s)

Rh2s = []
for dlta in Deltas:
    Rh2s.append(I_h2(dlta, eps))
Rh2s = np.array(Rh2s)

Rh_opt_ds = []
for dlta in Deltas:
    Rh_opt_ds.append(I_opt(dlta, 0.1, eps = eps))
Rh_opt_ds = np.array(Rh_opt_ds)


colors = ['darkred',"steelblue", 'darkblue','gray','darkorange',"lightgreen", 'green', "darkgreen"]
linestyles = ["-", "-.", "--", ":", "-.", ":", ":", ":"]
different_s = [1, 1.5, 2, "log", "ars", "opt-0.3", "opt-0.2", "opt-0.1"]


ax[0].plot(Deltas, Rh_opt, label=r"${\mathrm{Tr}}$-$\mathrm{GoF}$", linestyle="-", color="red")
ax[0].plot(Deltas, Rh1s, label=r"$h_{\mathrm{ars}}$", linestyle="-.", color="blue")
ax[0].plot(Deltas, Rh2s, label=r"$h_{\mathrm{log}}$", linestyle=":", color="gray")
ax[0].plot(Deltas, compute_h3_R_new(0.5, eps), label=r"$h_{\mathrm{ind},0.5}$", linestyle=":", color="lightgrey")
ax[0].plot(Deltas, Rh_opt_ds, label=r"$h_{\mathrm{opt}, 0.1}$", linestyle="--", color="black")

ax[0].set_ylim(ymin=1e-4)
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[0].set_xticks([1e-3, 1e-2, 1e-1, 1],labels=[r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
ax[0].set_xlabel(r"$\Delta$")
ax[0].set_ylabel(r"$R_{\mathcal{P}_{\Delta}}(\mathrm{detection~rule})$")
ax[0].text(0.05, 0.95, rf'$\varepsilon=1$', transform=ax[0].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')
print("For the second figure......")


eps = 0.5
Rh_opt = []
for dlta in Deltas:
    Rh_opt.append(for_a_delta(dlta, eps))
Rh_opt = np.array(Rh_opt)

Rh1s = []
for dlta in Deltas:
    Rh1s.append(I_h1(dlta, eps))
Rh1s = np.array(Rh1s)

Rh2s = []
for dlta in Deltas:
    Rh2s.append(I_h2(dlta, eps))
Rh2s = np.array(Rh2s)

Rh_opt_ds = []
for dlta in Deltas:
    Rh_opt_ds.append(I_opt(dlta, 0.1, eps = eps))
Rh_opt_ds = np.array(Rh_opt_ds)

ax[1].plot(Deltas, Rh_opt, label=r"${\mathrm{Tr}}$-$\mathrm{GoF}$", linestyle="-", color="red")
ax[1].plot(Deltas, Rh1s, label=r"$h_{\mathrm{ars}}$", linestyle="-.", color="blue")
ax[1].plot(Deltas, Rh2s, label=r"$h_{\mathrm{log}}$", linestyle=":", color="gray")
ax[1].plot(Deltas, compute_h3_R_new(0.5, eps), label=r"$h_{\mathrm{ind},0.5}$", linestyle=":", color="lightgrey")
ax[1].plot(Deltas, Rh_opt_ds, label=r"$h_{\mathrm{opt}, 0.1}$", linestyle="--", color="black")


ax[1].set_ylim(ymin=1e-4)
ax[1].set_yscale('log')
ax[1].set_xscale('log')
ax[1].set_xticks([1e-3, 1e-2, 1e-1, 1],labels=[r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
ax[1].set_xlabel(r"$\Delta$")
# ax[1].set_ylabel(r"$R_{\mathcal{P}_{\Delta}}(\mathrm{detection~rule})$")
ax[1].text(0.05, 0.95, rf'$\varepsilon=0.5$', transform=ax[1].transAxes, fontsize=16, verticalalignment='top',  horizontalalignment='left')


plt.legend(bbox_to_anchor =(1.05,1), loc='upper left', borderaxespad=0., frameon=False)
# plt.legend()
plt.tight_layout()
plt.savefig(f'5efficiency/efficiency-robust.pdf', dpi=300)
