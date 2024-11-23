import torch
import numpy as np


def substitution_attack(tokens,p,vocab_size,distribution=None):
    if p == 0:
        return tokens
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1).flatten()
    tokens[idx] = samples[idx]
    
    return tokens


def substitution_con_attack(tokens,p,vocab_size,distribution=None,block=3):
    assert block >= 1
    if p == 0:
        return tokens
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size

    used_N = int(p*len(tokens))
    used_block = np.random.randint(low=1,high=block)
    
    idx = torch.randperm(len(tokens))[:used_N]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1).flatten()
    tokens[idx] = samples[idx]
    
    return tokens


def deletion_attack(tokens,p):
    if p == 0:
        return tokens
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    keep = torch.ones(len(tokens),dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep]
        
    return tokens

def insertion_attack(tokens,p,vocab_size,distribution=None):
    if p == 0:
        return tokens
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i],samples[i],tokens[i:]])
        tokens[i] = samples[i]
        
    return tokens
