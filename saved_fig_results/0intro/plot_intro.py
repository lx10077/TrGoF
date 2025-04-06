#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def from_s2label(s):
    if s == "log":
        return r"$h_{\mathrm{log}}$"
    elif s == "ars":
        return r"Aaronson [1]"
    elif s == "opt-0.1" or s == "opt-0.2" or s == "opt-0.3":         
        return r"Li et al. [48]"
    elif s == 2 or s == 1 or s == 1.5:
        return r"$\mathrm{Tr}$-$\mathrm{GoF}$"
    else:
        return rf"$s={s}$"
    
def from_s2color(s):
    if s == "ars":
        return "-.", 'blue'
    elif type(s) is str and "opt-" in s:
        return "--", 'black'
    elif s == 2 or s== 1.5 or s==1:
        return "-", "red"
    elif s == "log":
        return ":", 'gray'

alpha = 0.01
final_result = dict()
temp = 0.3
corrupt = 0.05
num = 0.7
mask = True

exp_name1 = f"0intro/1p3B-parafinal-c5-m400-T1000-temp{temp}-maskTrue-alpha0.01-nsiuwm-0.7-figdata.pkl"
final_result = pickle.load(open(exp_name1, "rb"))
result = final_result["0"]
corrup_result = final_result[f"{corrupt}"]

save_dir = f"0intro/1p3B-LLMNoLarge-c5-m400-T1000-alpha0.01-temp{temp}-{mask}-nsiuwm-{num}.pkl"
results_dict = pickle.load(open(save_dir, "rb"))
raw_data = results_dict["H1-watermark"]
corrp_data = results_dict[corrupt*400]

## First Image
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5), sharey=True)

first = 400
for i, s in enumerate(["ars", "opt-0.1"]):
    ls, cl = from_s2color(s)
    ax.plot(raw_data[s]["x"][:first], np.array(raw_data[s]["y"][:first]), linestyle=ls, color=cl,label=from_s2label(s))

for i, s in enumerate(["ars", "opt-0.05"]):
    ls, cl = from_s2color(s)
    ax.plot(corrup_result[s]["x"][:first], np.array(corrup_result[s]["y"][:first]), linestyle=ls, color=cl, linewidth=1)

for i, s in enumerate(["ars", "opt-0.1"]):
    ls, cl = from_s2color(s)
    ax.plot(corrp_data[s]["x"][:first], np.array(corrp_data[s]["y"][:first]), linestyle=ls, color=cl, linewidth=0.5)

a1, b1, c1 = (400, raw_data["ars"]["y"][400-1]), (400, corrup_result["ars"]["y"][400-1]), (400, corrp_data["ars"]["y"][400-1])  # First arrow from point a to b
print("Drops from", a1[-1], " to ", b1[-1], "to", c1[-1])

a3, b3, c3 = (350, raw_data["ars"]["y"][350-1]), (350, corrup_result["ars"]["y"][350-1]), (350, corrp_data["ars"]["y"][350-1])  # First arrow from point a to b
print("Drops from", a3[-1], " to ", b3[-1], "to", c3[-1])


a2, b2, c2 = (350, raw_data["opt-0.1"]["y"][350-1]), (350, corrup_result["opt-0.05"]["y"][350-1]), (350, corrup_result["opt-0.1"]["y"][350-1]) # Second arrow from another point a to b
print("Drops from", a2[-1], " to ", b2[-1], "to", c2[-1])

plt.annotate('', xy=b1, xytext=a1,
             arrowprops=dict(arrowstyle="fancy",facecolor='red', edgecolor="none", linewidth=4))

plt.text(b1[0] - 130, (b1[1] + a1[1]) / 2, 'Paraphrase edits', fontsize=14, color='red')

plt.annotate('', xy=c1, xytext=b1,
             arrowprops=dict(arrowstyle="fancy", facecolor='red', edgecolor="none",linewidth=4))

plt.text(c1[0] - 130, (c1[1] + b1[1]) / 2, 'Adversarial edits', fontsize=14, color='red')


ax.set_ylabel('Detection power')
ax.set_xlabel('Text length')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"0intro/intro-{temp}.pdf", dpi=300)

## Second Image
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4), sharey=True)
first = 400
different_s = [1.5, "ars", "opt-0.1"]
for i, s in enumerate(different_s):
    ls, cl = from_s2color(s)
    x =  np.insert(np.array(raw_data[s]["x"]), 0, 0)
    y =  np.insert(np.array(raw_data[s]["y"]), 0, 0)
    ax[0].plot(x,y , linestyle=ls, color=cl, label=from_s2label(s))
    ax[0].set_title("Without human edit")

ax[0].set_ylabel('Detection power')
ax[0].set_xlabel('Text length')

first = 400
for i, s in enumerate([1.5, "ars", "opt-0.05"]):
    ls, cl = from_s2color(s)
    ax[1].plot(corrup_result[s]["x"][:first], np.array(corrup_result[s]["y"][:first]), linestyle=ls, color=cl, label=from_s2label(s))
    ax[1].set_title(rf"$5\%$ paraphrase edits")
ax[1].set_xlabel('Text length')


first = 400
for i, s in enumerate(different_s):
    ls, cl = from_s2color(s)
    ax[2].plot(corrp_data[s]["x"][:first], np.array(corrp_data[s]["y"][:first]), linestyle=ls, color=cl, label=from_s2label(s))
    ax[2].set_title(rf"$5\%$ adversarial edits")
ax[2].set_xlabel('Text length')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"0intro/intro-para-{temp}.pdf", dpi=300)
