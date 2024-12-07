#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from sampling import  gumbel_key_func, gumbel_Y
import torch
import argparse
import gc
import os

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="gumbel",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
parser.add_argument('--seed',default=1,type=int)
parser.add_argument('--temp',default=0.8, type=float)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--N',default=11,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language',default="french",type=str)
parser.add_argument('--truncate_vocab',default=8,type=int)

parser.add_argument('--all_temp', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7], help="A list of temperatures used to generate watermarked texts.")
parser.add_argument('--substitution', action='store_true', help="If set, substitution will be True; otherwise, it defaults to False.")
parser.add_argument('--deletion', action='store_true', help="If set, deletion will be True; otherwise, it defaults to False.")
parser.add_argument('--insertion', action='store_true', help="If set, insertion will be True; otherwise, it defaults to False.")
parser.add_argument('--translation', action='store_true', help="If set, translation will be True; otherwise, it defaults to False.")

args = parser.parse_args()
args.rt_translate = True

print(args.model)

if args.model == "facebook/opt-1.3b":
    size = "1p3"
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    size = "2p7"
else:
    raise ValueError

key = args.seed
torch.manual_seed(key)
seed_key = 15485863
segment = args.N
c = args.c
T = args.T
N = args.N

compute_sub = args.substitution
compute_del = args.deletion
compute_ist = args.insertion
rt_translate = args.translation

used_T = args.T
all_temperatures = args.all_temp

if size == "1p3":
    vocab_size = 50272
elif size == "2p7":
    vocab_size = 32000

for latter in [f"nsiuwm-{args.non_wm_temp}"]:

    print(latter)

    def get_dir(temp, task="sub"):
        previous_name = f"raw_data/{size}B-Gumbel-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-15485863-temp{temp}-{latter}.pkl"
        exp_name = f"corrupted_data/{size}B-robust-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-{N}-temp{temp}-{latter}"
        exp_name11=f"result/{size}B-robust-c{c}-m{args.m}-T{used_T}-{args.seed_way}-{key}-temp{temp}-{task}-{latter}-{N}.pkl"
        os.makedirs(os.path.dirname(exp_name11), exist_ok=True)
        return previous_name, exp_name, exp_name11

    import pickle

    def compute_Ys(A, corrupted_data, prompts, is_null=False):
        cor_level, T, used_m =  corrupted_data.shape

        full_Ys = []
        for k in tqdm(range(cor_level)):
            computed_Ys = []
            for i in range(T):
                if i % 100 == 0:
                    print(i)
                if not is_null:
                    text = corrupted_data[k][i]
                    prompt = prompts[i]
                    full_texts =  torch.cat([prompt[-c:],text])
                else:
                    full_texts =  corrupted_data[k][i]
                # print("text", text.shape, "promt", prompt.shape, "full",full_texts.shape)

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


    def compute_sub_for_temp(temp):
        previous_name, exp_name, exp_name11 = get_dir(temp, "sub")
        results1 = pickle.load(open(exp_name + "-sub.pkl", "rb"))
        print(exp_name)
        print(results1.keys())

        results1000 = pickle.load(open(previous_name, "rb"))
        prompts = results1000['prompts']
        print(results1000.keys())

        generator = torch.Generator()
        A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, seed_key, c, args.seed_way)
        Ys_dict = dict()

        sub_dict = results1["sub"]["curupt_water"]
        sub_Ys = compute_Ys(A, sub_dict, prompts)
        Ys_dict["sub"] = sub_Ys.tolist()
        pickle.dump(Ys_dict, open(exp_name11, 'wb'))

        torch.cuda.empty_cache()
        gc.collect()

    if compute_sub:
        print("Compute the pivotal statistics for random substitution...")
        for temp in all_temperatures:
            compute_sub_for_temp(temp)


    def compute_del_for_temp(temp):
        previous_name, exp_name, exp_name11 = get_dir(temp, "dlt")
        results1 = pickle.load(open(exp_name + "-dlt.pkl", "rb"))
        print(exp_name)
        print(results1.keys())

        results1000 = pickle.load(open(previous_name, "rb"))
        prompts = results1000['prompts']
        print(results1000.keys())

        generator = torch.Generator()
        A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, seed_key, c, args.seed_way)
        Ys_dict = dict()

        dlt_dict = results1["dlt"]["curupt_water"]     
        dlt_Ys = compute_Ys(A, dlt_dict, prompts)
        Ys_dict["dlt"] = dlt_Ys.tolist()
        pickle.dump(Ys_dict, open(exp_name11, 'wb'))

        torch.cuda.empty_cache()
        gc.collect()


    if compute_del:
        print("Compute the pivotal statistics for random deletion...")
        for temp in all_temperatures:
            compute_del_for_temp(temp)


    def compute_ist_for_temp(temp):
        previous_name, exp_name, exp_name11 = get_dir(temp, "ist")
        results1 = pickle.load(open(exp_name + "-ist.pkl", "rb"))
        print(exp_name)
        print(results1.keys())

        results1000 = pickle.load(open(previous_name, "rb"))
        prompts = results1000['prompts']
        print(results1000.keys())

        generator = torch.Generator()
        A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, seed_key, c, args.seed_way)
        Ys_dict = dict()

        ist_dict = results1["ist"]["curupt_water"]
        ist_Ys = compute_Ys(A, ist_dict, prompts)
        Ys_dict["ist"] = ist_Ys.tolist()
        pickle.dump(Ys_dict, open(exp_name11, 'wb'))

        torch.cuda.empty_cache()
        gc.collect()


    if compute_ist:
        print("Compute the pivotal statistics for random insertion...")
        for temp in all_temperatures:
            compute_ist_for_temp(temp)


    if rt_translate:
        print("Compute the pivotal statistics for roundtrip translation...")
        def get_trans_dir(temp):
            previous_name = f"raw_data/{size}B-Gumbel-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-15485863-temp{temp}-{latter}.pkl"
            exp_name = f"corrupted_data/{size}B-robust-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-{N}-temp{temp}-{latter}"
            exp_name11 = f"result/{size}B-robust-c{c}-m{args.m}-T{used_T}-{args.seed_way}-{key}-temp{temp}-{args.language}-trans-{latter}.pkl"
            os.makedirs(os.path.dirname(exp_name11), exist_ok=True)
            return previous_name, exp_name, exp_name11

        for temp in all_temperatures:
            previous_name, exp_name, exp_name11 = get_trans_dir(temp)
            results1 = pickle.load(open(exp_name + f"-trans-{args.language}.pkl", "rb"))

            results1000 = pickle.load(open(previous_name, "rb"))
            prompts = results1000['prompts']

            generator = torch.Generator()
            A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, seed_key, c, args.seed_way)
            Ys_dict = dict()

            sub_dict = results1["trans"]["curupt_water"]
            sub_Ys = compute_Ys(A, sub_dict, prompts)
            Ys_dict["trans"] = sub_Ys.tolist()
            pickle.dump(Ys_dict, open(exp_name11, 'wb'))

            torch.cuda.empty_cache()
            gc.collect()

    print("Finish computing and saving Y for", latter)
