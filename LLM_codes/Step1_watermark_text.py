from time import time
import os
import torch
from generation import WatermarkGenerate
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="Gumbel",type=str)
parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--batch_size',default=10,type=int)
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)
parser.add_argument('--T',default=1000,type=int)
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--truncate_vocab',default=8,type=int)
args = parser.parse_args()
print(args)

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
print(f"Using {torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model)

T = args.T                                    # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
new_tokens = args.m                           # number of tokens to generate
load_local_data = True
buffer_tokens = args.buffer_tokens 
prompt_tokens = args.prompt_tokens            # minimum prompt length
non_wm_temp = args.non_wm_temp                # temperature used by non-watermarked texts


def generate_text_for_a_T(temp):    

    if load_local_data:
        # Load local data
        with open('c4/c4.json', 'r') as f:
            lines = f.readlines()
        ds_iterator = iter(json.loads(line) for line in lines)
    else:
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True, cache_dir="/dbfs/")
        ds_iterator = iter(dataset)

    print(temp)
    results = defaultdict(dict)
    results['args'] = copy.deepcopy(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    eff_vocab_size = vocab_size - args.truncate_vocab
    print("Emebebding:", vocab_size)
    print(f'Loaded the model (t = {time()-t0} seconds)')

    WG = WatermarkGenerate(model, 
                        vocab_size=vocab_size, 
                        key=args.seed,
                        text_length=args.m, 
                        watermark_type=args.method, 
                        temperature=temp, 
                        text_window=args.c, 
                        seeding_scheme=args.seed_way,
                        non_wm_temp=non_wm_temp)

    t1 = time()

    ## Get T prompts which length is truncated to m
    prompts = []
    itm = 0
    while itm < T:
        example = next(ds_iterator)
        text = example['text']

        tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
        if len(tokens) < prompt_tokens + new_tokens:
            continue
        prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
        prompts.append(prompt)

        itm += 1
    prompts = torch.vstack(prompts)
    results['prompts'] = copy.deepcopy(prompts)
    pickle.dump(results, open("C4-prompts.pkl", "wb"))

     ## Start getting generated tokens and pseudo-random variables.
    watermarked_samples = []
    generated_Ys = []
    generated_top_probs = []
    all_where_watermarks = []
    for batch in tqdm(range(n_batches)):
        idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))
        generated_tokens, Ys, top_probs, where_watermarks = WG(prompts[idx],1.)
        watermarked_samples.append(generated_tokens[:,prompt_tokens:])  # Shape (Batch_size, new_tokens)
        generated_Ys.append(Ys)  # Shape (Batch_size, new_tokens)
        generated_top_probs.append(top_probs)  # Shape (Batch_size, new_tokens)
        all_where_watermarks.append(where_watermarks) # Shape (Batch_size, new_tokens)

    watermarked_samples = torch.cat(watermarked_samples, axis=0)
    generated_Ys = torch.cat(generated_Ys, axis=0)
    generated_top_probs = torch.cat(generated_top_probs, axis=0)
    all_where_watermarks = torch.cat(all_where_watermarks, axis=0)

    results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
    results['watermark']['Ys'] = copy.deepcopy(generated_Ys)
    results['watermark']['top_probs'] = copy.deepcopy(generated_top_probs)
    results['watermark']['where_watermark'] = copy.deepcopy(all_where_watermarks)

    print(f'Generated samples in (t = {time()-t1} seconds)')

    if args.model == "facebook/opt-1.3b":
        model_name = "1p3B"
    elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
        model_name = "2p7B"
    else:
        raise ValueError(f"No such model name!: {args.model}.")
    # nsiuwm = no sclae in un watermark
    exp_name = f"raw_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{temp}-nsiuwm-{non_wm_temp}.pkl"
    os.makedirs(os.path.dirname(exp_name), exist_ok=True)
    pickle.dump(results,open(exp_name,"wb"))

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    for temp in [0.1, 0.3, 0.5, 0.7, 1]:
        generate_text_for_a_T(temp)
