from time import time
import os
import torch
import gc
from transformers import AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np

from attacks import substitution_attack,insertion_attack,deletion_attack

import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="Gumbel",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
parser.add_argument('--seed',default=1,type=int)
parser.add_argument('--temp',default=1, type=float)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--batch_size',default=10,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--N',default=11,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--language',default="french",type=str)
parser.add_argument('--truncate_vocab',default=8,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)

parser.add_argument('--all_temp', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7], help="A list of temperatures used to generate watermarked texts.")
parser.add_argument('--substitution', action='store_true', help="If set, substitution will be True; otherwise, it defaults to False.")
parser.add_argument('--deletion', action='store_true', help="If set, deletion will be True; otherwise, it defaults to False.")
parser.add_argument('--insertion', action='store_true', help="If set, insertion will be True; otherwise, it defaults to False.")
parser.add_argument('--translation', action='store_true', help="If set, translation will be True; otherwise, it defaults to False.")

args = parser.parse_args()

## We only corrupt the data.
compute_sub = args.substitution
compute_del = args.deletion
compute_ist = args.insertion
rt_translate1 = args.translation


latter = f"nsiuwm-{args.non_wm_temp}"  ## suffix for the generated texts
results = defaultdict(dict)
results['args'] = copy.deepcopy(args)
print(args)
print(latter)

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)

if args.model == "facebook/opt-1.3b":
    vocab_size = 50272
    model_prefix = "1p3"
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    model_prefix = "2p7"
    vocab_size = 32000
model_name = model_prefix + "B"

tokenizer = AutoTokenizer.from_pretrained(args.model)
eff_vocab_size = vocab_size - args.truncate_vocab
print(eff_vocab_size)
print(f'Loaded the model (t = {time()-t0} seconds)')

T = args.T                                    # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
print("continue")


def corrupt(tokens, substitution, deletion, insertion, ):
    # The deletion, insertion, substitution is a propotion
    tokens = substitution_attack(tokens,substitution,eff_vocab_size)
    tokens = deletion_attack(tokens,deletion)
    tokens = insertion_attack(tokens,insertion,eff_vocab_size)
    return tokens


def get_currpted_data(watermarked_samples, substitution, deletion, insertion, new_tokens):
    watermarked_samples = torch.clip(watermarked_samples,max=eff_vocab_size-1)

    corrupted_watermark_data = []
    corrupted_watermark_data_text = []
    for itm in tqdm(range(args.T),position=0,leave=True):

        # Corrupted the watermarked data
        watermarked_sample_pre = watermarked_samples[itm]
        watermarked_sample = corrupt(watermarked_sample_pre, substitution, deletion, insertion)
        watermarked_sample_text = tokenizer.decode(watermarked_sample, skip_special_tokens=True)
        if args.rt_translate:
            watermarked_sample_text = rt_translate(watermarked_sample_text)
        watermarked_sample = tokenizer.encode(watermarked_sample_text,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=2048)[0]
        if len(watermarked_sample) <= new_tokens :
            watermarked_sample = torch.nn.functional.pad(watermarked_sample,(new_tokens-len(watermarked_sample),0),"constant",0)
        else:
            watermarked_sample = watermarked_sample[:new_tokens]
        corrupted_watermark_data.append(watermarked_sample)
        corrupted_watermark_data_text.append(watermarked_sample_text)
    
    corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
    return corrupted_watermark_data.unsqueeze(0), corrupted_watermark_data_text


def define_range():
    return tqdm(np.linspace(0, (args.N-1)*0.05, args.N))


all_temperatures = args.all_temp

def get_name(temperature):
    name = f"raw_data/{model_prefix}B-Gumbel-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-15485863-temp{temperature}-{latter}.pkl"
    exp_name = f"corrupted_data/{model_name}-robust-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-{args.N}-temp{temperature}-{latter}"
    return name, exp_name

if compute_sub:
    print("Start to randomly substitute the text...\n")

    for temp in all_temperatures:
        name, exp_name = get_name(temp)
        print(temp, name)

        results = pickle.load(open(name, "rb"))
        watermarked_samples = results["watermark"]["tokens"]

        corrupted_dataset = defaultdict(dict)
        print()
        corrupted_watermark_data = []
        corrupted_watermark_data_text = []
        for sub in define_range():
            corrupted_watermark_data0,corrupted_watermark_data_text0 = get_currpted_data(watermarked_samples, sub, 0, 0, new_tokens=args.m)
            corrupted_watermark_data.append(corrupted_watermark_data0)
            corrupted_watermark_data_text.append(corrupted_watermark_data_text0)

        corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
        corrupted_dataset["sub"]["curupt_water"] = copy.deepcopy(corrupted_watermark_data)
        corrupted_dataset["sub"]["curupt_water_text"] = corrupted_watermark_data_text

        os.makedirs(os.path.dirname(exp_name+"-sub.pkl"), exist_ok=True)
        pickle.dump(corrupted_dataset,open(exp_name+"-sub.pkl","wb"))

        torch.cuda.empty_cache()
        gc.collect()


if compute_del:
    print("Start to randomly delete the text...\n")
    for temp in all_temperatures:
        name, exp_name = get_name(temp)
        print(temp, name)

        results = pickle.load(open(name, "rb"))
        # prompts = results['prompts']
        watermarked_samples = results["watermark"]["tokens"]

        corrupted_dataset = defaultdict(dict)
        print()
        corrupted_watermark_data = []
        corrupted_watermark_data_text = []
        for dlt in define_range():
            corrupted_watermark_data0,corrupted_watermark_data_text0 = get_currpted_data(watermarked_samples, 0, dlt, 0, new_tokens=int(0.5*args.m))
            corrupted_watermark_data.append(corrupted_watermark_data0)
            corrupted_watermark_data_text.append(corrupted_watermark_data_text0)

        corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
        corrupted_dataset["dlt"]["curupt_water"] = copy.deepcopy(corrupted_watermark_data)
        corrupted_dataset["dlt"]["curupt_water_text"] = corrupted_watermark_data_text

        os.makedirs(os.path.dirname(exp_name+"-dlt.pkl"), exist_ok=True)
        pickle.dump(corrupted_dataset,open(exp_name+"-dlt.pkl","wb"))

        torch.cuda.empty_cache()
        gc.collect()

if compute_ist:
    print("Start to randomly insert the text...\n")
    for temp in all_temperatures:
        name, exp_name = get_name(temp)
        print(temp, name)

        results = pickle.load(open(name, "rb"))
        watermarked_samples = results["watermark"]["tokens"]

        corrupted_dataset = defaultdict(dict)
        print()
        corrupted_watermark_data = []
        corrupted_watermark_data_text = []
        for ist in define_range():
            corrupted_watermark_data0,corrupted_watermark_data_text0 = get_currpted_data(watermarked_samples, 0, 0, ist, new_tokens=args.m)
            corrupted_watermark_data.append(corrupted_watermark_data0)
            corrupted_watermark_data_text.append(corrupted_watermark_data_text0)

        corrupted_watermark_data = torch.vstack(corrupted_watermark_data)
        corrupted_dataset["ist"]["curupt_water"] = copy.deepcopy(corrupted_watermark_data)
        corrupted_dataset["ist"]["curupt_water_text"] = corrupted_watermark_data_text

        os.makedirs(os.path.dirname(exp_name+"-ist.pkl"), exist_ok=True)
        pickle.dump(corrupted_dataset,open(exp_name+"-ist.pkl","wb"))

        torch.cuda.empty_cache()
        gc.collect()


if rt_translate1:
    print("Start to perform roundtrip translation...\n")

    for temp in all_temperatures:
        name, exp_name = get_name(temp)

        if args.language == "french":
            en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
            en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
            en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name).to(device)

            ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
            ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
            ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name).to(device)

        elif args.language == "russian":
            en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
            en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
            en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name).to(device)

            ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
            ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
            ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name).to(device)
        else:
            raise

        def rt_translate(text):
            try:
                tokens = en_ne_tokenizer(text.split('. '), return_tensors="pt", padding=True).to(device)
                tokens = en_ne_model.generate(**tokens,max_new_tokens=450)
                french_text = ' '.join([en_ne_tokenizer.decode(t, skip_special_tokens=True) for t in tokens])

                tokens = ne_en_tokenizer(french_text.split('. '), return_tensors="pt", padding=True).to(device)
                tokens = ne_en_model.generate(**tokens,max_new_tokens=500)
                roundtrip_text = ' '.join([ne_en_tokenizer.decode(t, skip_special_tokens=True) for t in tokens])
            except:
                roundtrip_text = ""
                print("print zero")

            return roundtrip_text
        
        results = pickle.load(open(name, "rb"))
        watermarked_samples = results["watermark"]["tokens"]

        corrupted_dataset = defaultdict(dict)
        corrupted_watermark_data,corrupted_watermark_data_text = get_currpted_data(watermarked_samples, 0, 0, 0, new_tokens=args.m)

        corrupted_dataset["trans"]["curupt_water"] = copy.deepcopy(corrupted_watermark_data)
        corrupted_dataset["trans"]["curupt_water_text"] = corrupted_watermark_data_text
        
        output_path = exp_name + f"-trans-{args.language}.pkl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(corrupted_dataset, f)

        torch.cuda.empty_cache()
        gc.collect()
