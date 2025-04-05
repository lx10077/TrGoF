#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

print()
c=5
import pickle
print(c, "\n", )

s_lst = [1, 1.5, 2, "log","ars",  "opt-0.3", "opt-0.2", "opt-0.1"]
latter=""
print(s_lst)

for size in ["1p3", "2p7"]:
    print("==========================>")
    print(f"The model is {size} B....")
    for creat in [True, False]:
        for task in ["sub", "dlt", "ist"]:
            print("Creative generation:", creat)
            print("The task is:", task)
            temp = 1
            save_dir = f"{size}B-creat{creat}-c{c}-m400-T100-temp{temp}-alpha0.01-True-{task}{latter}.pkl"
            save_dict = pickle.load(open(save_dir, "rb"))
            result_lst = []

            for s  in s_lst:
                v = save_dict[s]
                f = round(np.mean(v)/400 * 100,2)
                result_lst.append(f)
            
            largest = np.max(result_lst)
            result = "&".join([f"\\textbf{{{x}}}" if x == largest else str(x) for x in result_lst])

            print(result)
        print()
