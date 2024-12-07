#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import gamma, norm
import torch
import argparse
import pickle
import copy
from collections import defaultdict
from transformers import AutoTokenizer


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="Gumbel",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
parser.add_argument('--seed',default=1,type=int)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--truncate_vocab',default=8,type=int)
parser.add_argument('--all_temp', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7], help="A list of temperatures used to generate watermarked texts.")
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--alpha',default=0.01,type=float)

args = parser.parse_args()

key = args.seed
torch.manual_seed(key)
seed_key = 15485863
c = args.c
T = args.T
m = args.m
mask = True
temperatures = args.all_temp
latter = f"nsiuwm-{args.non_wm_temp}"
alpha = args.alpha

if args.model == "facebook/opt-1.3b":
    size = "1p3"
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    size = "2p7"
else:
    raise ValueError(f"No name for this model: {args.model}! Currently give name to either facebook/opt-1.3b or princeton-nlp/Sheared-LLaMA-2.7B.")


def compute_score(Ys, s=2,eps=1e-10, mask=mask):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps, axis=-1)
    m = ps.shape[-1]
    first = m
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
    
    return m*np.max(final,axis=-1)

print("Begining ploting")

def compute_quantile(m, alpha, s, mask=True):
    qs = []
    for _ in range(10):
        raw_data = np.random.uniform(size=(10000, m))
        H0s = compute_score(raw_data, s=s, mask=mask)
        log_H0s = np.log(H0s+1e-10)
        q = np.quantile(log_H0s, 1-alpha)
        qs.append(q)
    return np.mean(qs,axis=0)


def HC_for_a_given_fraction(Ys, ratio, alpha=0.01, s=2, mask=True):
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
    

def h_opt_gum(Ys, delta0=0.2, alpha=0.01):
    # Compute critical values
    Ys = np.array(Ys)
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



def plot_length_on_axis(current_ax, Y, alpha, x_name, legend=True, H="H1", log=False):
    global different_s
    used_m = Y.shape[-1]
    print(H, Y.shape)

    x = np.arange(1,1+used_m)
    result_set = defaultdict(dict)
    j = -1
    start_point=3
    for s in different_s:
        j += 1
        if type(s) is not str:
            x = np.arange(1,1+used_m, 5)
            y = []
            for x_point in x:
                HC, log_critical_value = HC_for_a_given_fraction(Y, x_point, alpha, s,)
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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams.update({
    'font.size': 14,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


linestyles = ["-", "-.", "--","-.", ":", "--", "-.", ":","-.",]
colors = ["tab:blue", "tab:orange", "black", "tab:purple", "tab:red",  "tab:pink", "tab:gray",   "tab:brown", "tab:green", ]
results = dict()



def remove_repeated(Ys, filered_length=None,filter_data=False, mute=True):
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
        print("If we want at 1000 samples, there should >=", 1000/len(Ys))
        print(">=350:", np.mean(unique_elements_num>=350))
        print(">=300:", np.mean(unique_elements_num>=300))
        print(">=250:", np.mean(unique_elements_num>=250))
        print(">=200:", np.mean(unique_elements_num>=200))
        print(">=150:", np.mean(unique_elements_num>=150))
        print(">=100:", np.mean(unique_elements_num>=100))
        print(">=50:", np.mean(unique_elements_num>=50))

    if filered_length is not None:
        return filtered_rows
    
    if filter_data is True:
        # Find the indices of the 100 largest numbers
        largest_indices = np.argpartition(unique_elements_num, -1000)[-1000:]

        # Sort the indices by the actual values to get them in descending order
        largest_indices = largest_indices[np.argsort(-unique_elements_num[largest_indices])]

        # Get the largest 500 entries
        largest_values = unique_elements_num[largest_indices]

        print(np.array(largest_values)/Ys.shape[1])
        return Ys[largest_indices]
    return Ys


print()
print("Start plotting~!!")
different_s = ["log", "ars", 2, 1.5, 1, "opt-0.3", "opt-0.2", "opt-0.1"]
# temperatures = [0.1, 0.3, 0.5, 0.7, 1]

def replace_largest_k_with_uniform(Y, k, if_return_index=False):
    """
    Replace the largest k values in each row of a 2D numpy array Y with random values 
    from a uniform distribution on (0, 1).
    
    Parameters:
    Y (np.ndarray): 2D array of shape (n, d)
    k (int): Number of largest values in each row to replace

    Returns:
    np.ndarray: Modified array with the largest k values in each row replaced
    """
    n, d = Y.shape
    if k > d:
        raise ValueError("k should be less than or equal to the number of columns in Y")

    # Copy Y to avoid modifying the original array
    Y_modified = Y.copy()
    all_largest_k_indices = []
    
    for i in range(n):
        # Get indices of the largest k values in the row
        largest_k_indices = np.argpartition(Y[i], -k)[-k:]
        
        # Replace these values with random values from a uniform distribution on (0, 1)
        Y_modified[i, largest_k_indices] = np.random.uniform(0, 1, size=k)
        all_largest_k_indices.append(largest_k_indices)

    if if_return_index:
        return Y_modified, all_largest_k_indices
    
    else:
        return Y_modified





def find_nearest_indices(Ind, AInd):
    """
    For each index in Ind, find the nearest index in AInd.

    Parameters:
    Ind (np.ndarray): Array of indices for which we want to find nearest indices in AInd
    AInd (np.ndarray): Array of available replaceable indices

    Returns:
    np.ndarray: Array of the nearest indices in AInd for each index in Ind
    """
    nearest_indices = []

    for index in Ind:
        # Filter AInd to only include indices less than or equal to the current index
        valid_indices = AInd[AInd <= index]
        
        # If valid indices exist, choose the largest one (the closest previous index)
        if valid_indices.size > 0:
            nearest_index = valid_indices[-1]
        else:
            nearest_index = AInd[np.abs(AInd - index).argmin()]
        
        nearest_indices.append(nearest_index)
    
    return np.array(nearest_indices)


def adversarial_edit(Gum_Ys, watermarked_samples, vocab_size, top_k=40, new_tokens=None):
    if new_tokens is None:
        new_tokens = Gum_Ys.shape[1]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    Corruped_Y, all_largest_k_indices = replace_largest_k_with_uniform(Gum_Ys, top_k, if_return_index=True)
    eff_vocab_size = vocab_size - args.truncate_vocab
    watermarked_samples = torch.clip(watermarked_samples,max=eff_vocab_size-1)
    corrupted_watermark_data = []
    corrupted_watermark_data_text = []
    for itm in range(len(Corruped_Y)):
        watermarked_sample = watermarked_samples[itm]
        modify_index = all_largest_k_indices[itm]
        watermarked_sample[modify_index] = torch.randint(0, eff_vocab_size, (len(modify_index),))
        paraphsed_text = tokenizer.decode(watermarked_sample, skip_special_tokens=True)
        corrupted_watermark_data.append(watermarked_sample)
        corrupted_watermark_data_text.append(paraphsed_text)

    corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
    return corrupted_watermark_data.unsqueeze(0), corrupted_watermark_data_text


from sampling import  gumbel_key_func, gumbel_Y

def compute_Ys(A, corrupted_data, prompts, is_null=False):
    cor_level, T, used_m =  corrupted_data.shape

    full_Ys = []
    for k in tqdm(range(cor_level)):
        computed_Ys = []
        for i in range(T):
            if i % 100 == 1:
                print(i)
            if not is_null:
                text = corrupted_data[k][i]
                prompt = prompts[i]
                full_texts =  torch.cat([prompt[-c:],text])
            else:
                full_texts =  corrupted_data[k][i]

            this_Ys = []
            for j in range(used_m):
                given_seg = full_texts[:c+j].unsqueeze(0)
                xi,pi = A(given_seg)
                Y = gumbel_Y(full_texts[c+j].unsqueeze(0).unsqueeze(0), pi, xi)
                this_Ys.append(Y.unsqueeze(0))

            this_Ys = torch.vstack(this_Ys)
            computed_Ys.append(this_Ys.squeeze())
        computed_Ys = torch.vstack(computed_Ys).numpy()
        full_Ys.append(computed_Ys)
    return np.array(full_Ys)




if size == "1p3":
    vocab_size = 50272
else:
    vocab_size = 32000

for k, temp in enumerate(temperatures):
    print()
    print("Working on temperature:", temp)

    # From Gumbel-max
    Gum_name = f"raw_data/{size}B-Gumbel-c{c}-m{m}-T{T}-noncomm_prf-15485863-temp{temp}-{latter}.pkl"
    Gum_result = pickle.load(open(Gum_name, "rb"))
    Gum_Ys = np.array(Gum_result["watermark"]["Ys"])
    prompts = Gum_result["prompts"]
    repeated = np.array(Gum_result["watermark"]["where_watermark"])
    watermarked_samples = Gum_result["watermark"]["tokens"]

    save_dir = f"corrupted_data/{size}B-Gumbel-c{c}-m400-T1000-noncomm_prf-15485863-temp{temp}-{latter}-cor"

    print("Load Gum results...\n")
    print(Gum_Ys.shape)

    corruptions = [10, 20, 40, 80, 120]

    here_save_dir = save_dir+f"-all.pkl"  

    if not os.path.exists(here_save_dir):
        print("we generate the result!")

        os.makedirs(os.path.dirname(here_save_dir), exist_ok=True)

        corrupted_Y_dict = dict()
        corrupted_Y_dict[0] = Gum_Ys.tolist()
        
        for corrup in corruptions:
            corrup_dict= dict()
            generator = torch.Generator()
            A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, seed_key, c, args.seed_way)

            corrup_tokens, corrup_text = adversarial_edit(Gum_Ys, watermarked_samples, vocab_size, top_k=corrup, new_tokens=None)
            print("finish corrup")
            corrup_Y = compute_Ys(A, corrup_tokens, prompts)
            print("finish computing Y")

            corrupted_Y_dict[corrup] = corrup_Y.tolist()

            corrup_dict["corrup_tokens"] = copy.deepcopy(corrup_tokens)
            corrup_dict["corrup_text"] = corrup_text
            corrup_dict["corrup_Ys"] = corrup_Y.tolist()
            here_save_dir = save_dir+f"{corrup}.pkl"

            os.makedirs(os.path.dirname(here_save_dir), exist_ok=True)
            pickle.dump(corrup_dict,open(here_save_dir,"wb"))
        
        pickle.dump(corrupted_Y_dict, open(here_save_dir,"wb"))    
    else:
        print("we load the result!")
        with open(here_save_dir, "rb") as file:
            corrupted_Y_dict = pickle.load(file)


    fig, ax = plt.subplots(nrows=1, ncols=len(corruptions)+1, figsize=(4*(len(corruptions)+1),4))

    print("finish corruption and computation of Y.")
    
    if temp >= 0.5:
        use_log = True
    else:
        use_log = False

    H2set = plot_length_on_axis(ax[0], Gum_Ys, alpha, "Watermarked text length", legend=True, H="H1", log=use_log)
    final_result = dict()
    final_result["H1-watermark"] = H2set

    for l, each_corrup in enumerate(corruptions):
        corrup_Ys = np.array(corrupted_Y_dict[each_corrup])[0]
        ax[l+1].set_title(f"Replace {each_corrup}")
        H2set = plot_length_on_axis(ax[l+1], corrup_Ys, alpha, "Watermarked text length", legend=True, H="H1", log=use_log)
        final_result[each_corrup] = H2set

    notation = "LLMNoLarge"
    save_dir = f"cor_data/{size}B-{notation}-c{c}-m{m}-T{T}-alpha{alpha}-temp{temp}-{mask}-{latter}.pkl"
    dir_path = os.path.dirname(save_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    pickle.dump(final_result, open(save_dir, "wb"))

    plt.legend(bbox_to_anchor =(1.05,1), loc='upper left',borderaxespad=0.)
    plt.tight_layout()
    fig_save_dir = f"cor_fig/{size}B-{notation}-c{c}-m{m}-T{T}-alpha{alpha}-temp{temp}-{mask}-{latter}.pdf"
    dir_path = os.path.dirname(fig_save_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(fig_save_dir, dpi=300)
