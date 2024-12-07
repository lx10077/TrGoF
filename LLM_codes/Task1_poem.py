from time import time
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython import embed
from sampling import  gumbel_key_func, gumbel_Y
from scipy.stats import gamma
from tqdm import tqdm
from collections import defaultdict
import pickle
import copy
import gc
import numpy as np
from generation import WatermarkGenerate

import argparse

## We only generate the data.

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--method',default="Gumbel",type=str)
parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=5,type=int)
parser.add_argument('--batch_size',default=1,type=int) # batch size, how many prompts are used at the same time to generate texts.
parser.add_argument('--seed_way',default="noncomm_prf",type=str)
parser.add_argument('--m',default=400,type=int)  # The length of generated text
parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--temp',default=1,type=float)
parser.add_argument('--truncate_vocab',default=8,type=int)
parser.add_argument('--non_wm_temp',default=0.7,type=float)
parser.add_argument('--alpha',default=0.01,type=float)
args = parser.parse_args()
print(args)

alpha = args.alpha
temp = args.temp

poems_list = [
    'A Shropshire Lad by A.E. Housman',
    'A Valediction: Forbidding Mourning by John Donne',
    'Adonais by Percy Bysshe Shelley',
    'Aeneas and Dido by Christopher Marlowe',
    'Among School Children by W.B. Yeats',
    'An Essay on Man by Alexander Pope',
    'Annabel Lee by Edgar Allan Poe',
    'Aurora Leigh by Elizabeth Barrett Browning',
    'Beowulf by Anonymous (Old English)',
    'Childe Harold’s Pilgrimage by Lord Byron',
    'Christabel by Samuel Taylor Coleridge',
    'Do Not Go Gentle into That Good Night by Dylan Thomas',
    'Don Juan by Lord Byron',
    'Dover Beach by Matthew Arnold',
    'Elegy Written in a Country Churchyard by Thomas Gray',
    'Endymion by John Keats',
    'Evangeline by Henry Wadsworth Longfellow',
    'Fern Hill by Dylan Thomas',
    'Gerontion by T.S. Eliot',
    'Goblin Market by Christina Rossetti',
    'Home Thoughts, from Abroad by Robert Browning',
    'Howl by Allen Ginsberg',
    'Hyperion by John Keats',
    'Idylls of the King by Alfred Lord Tennyson',
    'Il Penseroso by John Milton',
    'In Memoriam by Alfred Lord Tennyson',
    'Kubla Khan by Samuel Taylor Coleridge',
    "L'Allegro by John Milton",
    'Lamia by John Keats',
    'Leaves of Grass by Walt Whitman',
    'Lycidas by John Milton',
    'Maud by Alfred Lord Tennyson',
    'Ode on a Grecian Urn by John Keats',
    'Ode to a Nightingale by John Keats',
    'Ode to the Confederate Dead by Allen Tate',
    'Ode to the West Wind by Percy Bysshe Shelley',
    'Ozymandias by Percy Bysshe Shelley',
    'Paradise Lost by John Milton',
    'Prometheus Unbound by Percy Bysshe Shelley',
    'Renascence by Edna St. Vincent Millay',
    'Resurrection by Boris Pasternak',
    'Sailing to Byzantium by W.B. Yeats',
    'Song of Myself by Walt Whitman',
    "Tam o' Shanter by Robert Burns",
    'The Aeneid by Virgil',
    'The Ballad of East and West by Rudyard Kipling',
    'The Ballad of Reading Gaol by Oscar Wilde',
    'The Bells by Edgar Allan Poe',
    'The Bells of Rhymney by Idris Davies',
    'The Bronze Horseman by Alexander Pushkin',
    'The Brook by Alfred Lord Tennyson',
    'The Canterbury Tales by Geoffrey Chaucer',
    'The Charge of the Light Brigade by Alfred Lord Tennyson',
    'The City of Dreadful Night by James Thomson',
    'The Cloud by Percy Bysshe Shelley',
    'The Courtship of Miles Standish by Henry Wadsworth Longfellow',
    'The Deserted Village by Oliver Goldsmith',
    'The Divine Comedy by Dante Alighieri',
    'The Dream of Gerontius by John Henry Newman',
    'The Dunciad by Alexander Pope',
    'The Eve of St. Agnes by John Keats',
    'The Faerie Queene by Edmund Spenser',
    'The Fall of Hyperion by John Keats',
    'The Faun by Jules Laforgue',
    'The Faun by Mallarmé',
    'The Flea by John Donne',
    'The Garden of Proserpine by Algernon Charles Swinburne',
    'The Golden Gate by Vikram Seth',
    'The Hollow Men by T.S. Eliot',
    'The Hunting of the Snark by Lewis Carroll',
    'The Iliad by Homer',
    'The Lady of Shalott by Alfred Lord Tennyson',
    'The Lotos-Eaters by Alfred Lord Tennyson',
    'The Love Song of J. Alfred Prufrock by T.S. Eliot',
    'The Odyssey by Homer',
    'The Poetry of Earth by John Keats',
    'The Prelude by William Wordsworth',
    'The Prisoner of Chillon by Lord Byron',
    'The Raven by Edgar Allan Poe',
    'The Rime of the Ancient Mariner by Samuel Taylor Coleridge',
    'The Rubaiyat of Omar Khayyam by Edward FitzGerald (Translation)',
    'The Scholar Gipsy by Matthew Arnold',
    'The Seafarer by Anonymous (Old English)',
    'The Second Coming by W.B. Yeats',
    'The Shepherd by William Blake',
    'The Song of Hiawatha by Henry Wadsworth Longfellow',
    'The Song of Roland by Anonymous',
    'The Song of the Wandering Aengus by W.B. Yeats',
    'The Tower by W.B. Yeats',
    'The Triumph of Life by Percy Bysshe Shelley',
    'The Tyger by William Blake',
    'The Vision of Piers Plowman by William Langland',
    'The Vision of Sir Launfal by James Russell Lowell',
    'The Wanderings of Oisin by W.B. Yeats',
    'The Waste Land by T.S. Eliot',
    'The Windhover by Gerard Manley Hopkins',
    'Thyrsis by Matthew Arnold',
    'Tintern Abbey by William Wordsworth',
    'Tithonus by Alfred Lord Tennyson',
    'To His Coy Mistress by Andrew Marvell'
]


T = len(poems_list)                                   # number of prompts/generations
n_batches = int(np.ceil(len(poems_list) / args.batch_size)) # number of batches
new_tokens = args.m                           # number of tokens to generate
buffer_tokens = args.buffer_tokens 
no_wm_temp = args.no_wm_temp
latter = f"-nsiuwm-{no_wm_temp}"

def f_opt(r, delta):
    inte_here = np.floor(1/(1-delta))
    rest = 1-(1-delta)*inte_here
    return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1)+1e-10)

def compute_score(Ys, mask, alpha=1., s=2,eps=1e-10):
    # assert -1 <= s <= 2
    ps = 1- Ys
    ps = np.sort(ps, axis=-1)
    m = ps.shape[-1]
    first = int(m* alpha)
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
    elif s > 0:
        final = (1-(rk**s)*(ps+eps)**(1-s)-((1-rk)**s)*((1-ps+eps)**(1-s)))/(s*(1-s))
    elif s == -1:
        final = (rk - ps)**2/(rk*(1-rk)+eps)/2
    else: # we must have -1 < s < 0
        final = (1-ps**(1-s)/(rk+eps)**(-s)-(1-ps)**(1-s)/(1-rk+eps)**(-s))/(s*(1-s))

    if mask:
        ind = (ps >= 1e-3)* (rk >= ps)
        final *= ind
        
    return m*np.max(final,axis=-1)


def compute_quantile(m, alpha, s, mask):
    # for _ in range(500):
    qs = []
    for _ in range(10):
        raw_data = np.random.uniform(size=(10000, m))
        H0s = compute_score(raw_data, s=s, mask=mask)
        log_H0s = np.log(H0s+1e-10)
        q = np.quantile(log_H0s, 1-alpha)
        qs.append(q)
    return np.mean(qs)

for used_model in ["facebook/opt-1.3b", "princeton-nlp/Sheared-LLaMA-2.7B"]:
# for used_model in [ "princeton-nlp/Sheared-LLaMA-2.7B"]:
    for creative_generation in [True, False]:

        results = defaultdict(dict)
        results['args'] = copy.deepcopy(args)

        # fix the random seed for reproducibility
        t0 = time()
        torch.manual_seed(args.seed)

        print(f"Using {torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(used_model)
        model = AutoModelForCausalLM.from_pretrained(used_model, device_map="auto", offload_folder="./offload_folder")
        model.eval()
        vocab_size = model.get_output_embeddings().weight.shape[0]
        eff_vocab_size = vocab_size - args.truncate_vocab
        print(f'Loaded the model (t = {time()-t0} seconds)', vocab_size)

        #/local_disk0/
        print("Successully loading dataset...")
        generate_data = True
        if used_model == "facebook/opt-1.3b":
            model_name = "1p3B"
        elif used_model == "princeton-nlp/Sheared-LLaMA-2.7B":
            model_name = "2p7B"
        else:
            raise ValueError

        prompts = []
        itm = 0
        for itm in range(len(poems_list)):
            this_poem = poems_list[itm]
            if creative_generation is True:
                text = f"Please write a new poem in the style of this one: {this_poem}.\n\n"
            else:
                text = f"Please recite the poem: {this_poem}.\n\n"

            prompt = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
            prompts.append(prompt)

        generate_data = not os.path.exists(f"raw_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{T}-{args.seed_way}-{args.seed}-creat{creative_generation}-temp{temp}{latter}.pkl")
        if generate_data:

            WG = WatermarkGenerate(model, 
                            vocab_size=vocab_size, 
                            key=args.seed,
                            text_length=args.m, 
                            watermark_type=args.method, 
                            temperature=temp, 
                            text_window=args.c, 
                            seeding_scheme=args.seed_way,
                            non_wm_temp=args.no_wm_temp)
            
            t1 = time()

            ## Start getting generated tokens and pseudo-random variables.
            watermarked_samples = []
            generated_Ys = []
            generated_top_probs = []
            all_where_watermarks = []
            for batch in tqdm(range(n_batches)):
                assert args.batch_size == 1

                idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))
                generated_tokens, Ys, top_probs, where_watermarks = WG(prompts[idx].unsqueeze(0), 1.)
                # print(batch, generated_tokens.shape, generated_tokens[:,-new_tokens:].shape, Ys.shape, top_probs.shape, where_watermarks.shape, )
                watermarked_samples.append(generated_tokens[:,-new_tokens:])  # Shape (Batch_size, new_tokens)
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

            exp_name = f"raw_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{T}-{args.seed_way}-{args.seed}-creat{creative_generation}-temp{temp}{latter}.pkl"
            os.makedirs(os.path.dirname(exp_name), exist_ok=True)
            pickle.dump(results,open(exp_name,"wb"))

            torch.cuda.empty_cache()
        else:
            exp_name = f"raw_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{T}-{args.seed_way}-{args.seed}-creat{creative_generation}-temp{temp}{latter}.pkl"
            results = pickle.load(open(exp_name,"rb"))
            watermarked_samples = results['watermark']['tokens']
            generated_Ys = results['watermark']['Ys']

        watermarked_samples = torch.clip(watermarked_samples,max=eff_vocab_size-1)

        print()

        def compute_Ys(A, prompt, text):
            # corrupted_data is just a sentence.
            used_m =  len(text)
            full_texts =  torch.cat([prompt[-args.c:],text])

            this_Ys = []
            for j in range(used_m):
                given_seg = full_texts[:args.c+j].unsqueeze(0)
                xi,pi = A(given_seg)
                Y = gumbel_Y(full_texts[args.c+j].unsqueeze(0).unsqueeze(0), pi, xi)
                this_Ys.append(Y.unsqueeze(0))

            this_Ys = torch.vstack(this_Ys).squeeze().numpy()
            return this_Ys

        def compute_critical_values(used_m, alpha, mask):
            criical_value = dict()
            for s in different_s:
                if s == "log":
                    criical_value[s] = -gamma.ppf(q=alpha,a=used_m)
                elif s == "ars":
                    criical_value[s] = gamma.ppf(q=1-alpha,a=used_m)
                elif type(s) == str and "opt" in s:
                    delta0 = float(s[4:])

                    def find_q(N=2500):
                        Null_Ys = np.random.uniform(size=(N, used_m))
                        Simu_Y = f_opt(Null_Ys, delta0)
                        Simu_Y = np.sum(Simu_Y, axis=1)/np.sqrt(used_m)
                        h_help_qs = np.quantile(Simu_Y, 1-alpha)
                        return h_help_qs
            
                    q_lst = []
                    for N in [2500] * 10:
                        q_lst.append(find_q(N))
                    criical_value[s] = np.mean(np.array(q_lst))
                elif type(s) == int or float:
                    criical_value[s] =  compute_quantile(used_m, alpha, s, mask)
                else:
                    raise ValueError(f"No such value of method: {s}.")
            return criical_value


        different_s = ["log", "ars", 2, 1.5, 1, "opt-0.3", "opt-0.2", "opt-0.1"]

        mask = True
        critical_value_100 = compute_critical_values(100, alpha, mask)
        critical_value_200 = compute_critical_values(200, alpha, mask)
        used_m_here = {"sub": (200,critical_value_200) , 
                    "dlt": (100,critical_value_100),
                    "ist": (200, critical_value_200)}


        for task in ["sub", "dlt", "ist"]:
        # for task in ["dlt"]:

            print("The current task is", task)
            result_dict = defaultdict(list)
            result_dict["method"] = different_s
            here_m, here_critical = used_m_here[task]
            print(here_m, here_critical)
            
            generator = torch.Generator()
            A = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, args.seed, args.c, args.seed_way)

            for itm in range(len(poems_list)):
                this_text = watermarked_samples[itm]
                this_prompt = prompts[itm]

                idx = torch.randperm(len(this_text))
                new_probs = torch.ones(size=(len(this_text),eff_vocab_size)) / eff_vocab_size
                new_samples = torch.multinomial(new_probs,1)
                if task == "sub":
                    new_samples = new_samples.flatten()

                def modifed_text(this_text, modifiy_type, modified_idx):
                    this_text = this_text.detach().clone()
                    if modifiy_type == "sub":
                        this_text[modified_idx] = new_samples[modified_idx]
                    elif modifiy_type == "dlt":
                        keep = torch.ones(len(this_text),dtype=torch.bool)
                        keep[modified_idx] = False
                        this_text = this_text[keep]
                    elif modifiy_type == "ist":
                        for i in modified_idx.sort(descending=True).values:
                            this_text = torch.cat([this_text[:i],new_samples[i],this_text[i:]])
                            this_text[i] = new_samples[i]
                    else:
                        raise ValueError(f"No such task: {task}.")
                    return this_text[:here_m]

                def detect(prompt, text, s, critical_value, mask):
                    this_Ys = compute_Ys(A, prompt, text)
                    if s == "log":
                        final = np.sum(np.log(this_Ys)) - critical_value[s]
                        return final, final >= 0
                    elif s == "ars":
                        final = np.sum(-np.log(1-this_Ys)) - critical_value[s]
                        return final, final >= 0
                    elif type(s) == str and "opt" in s:
                        delta0 = float(s[4:])
                        final = np.sum(np.sum(f_opt(this_Ys, delta0))/np.sqrt(len(this_Ys))-critical_value[s]) 
                        return final, final >= 0
                    elif type(s) == int or float:
                        final = np.log(compute_score(this_Ys, s=s, mask=mask)+1e-10) - critical_value[s]
                        return final, final >= 0
                    else:
                        raise ValueError(f"No such value of method: {s}.")

                for s in different_s:
                    l = 0
                    if task == "sub":
                        u = here_m-1
                    elif task == "dlt":
                        u = args.m-here_m-1
                    elif task == "ist":
                        u = args.m-1

                    while u- l >=2:
                        mid = (l+u)//2
                        new_text = modifed_text(this_text, task, idx[:mid])
                        final, detectable = detect(this_prompt, new_text, s, here_critical, mask)
                        if detectable:
                            l = mid
                        else:
                            u = mid
                    
                    if detectable:
                        result_dict[s].append(u)
                    else:
                        result_dict[s].append(l)

                    if itm % 10 == 1:
                        print(task, itm, s, "(u,l):", u, l, detectable, "final", final)

                print()

            save_dir = f"poem_result/{model_name}-creat{creative_generation}-c{args.c}-m{args.m}-T{T}-temp{temp}-alpha{alpha}-{mask}-{task}{latter}.pkl"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            pickle.dump(result_dict, open(save_dir, "wb"))

            torch.cuda.empty_cache()
            gc.collect()
