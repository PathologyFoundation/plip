import re

log_file_path = "/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning/Kather/train_ratio=1.0/PLIP_20230519-04.08.47_data=Kather_btch=128_wd=0.1_firstresize=512_pxsize=224_nepochs=10_validratio=0.1_optimizer=AdamW/random_seed=0/_training.log"
f1_weighted_scores = []

with open(log_file_path, "r") as log_file:
    for line in log_file:
        match = re.search(r"f1_weighted:\s+(\d+\.\d+)", line)
        if match:
            score = float(match.group(1))
            f1_weighted_scores.append(score)

print(f1_weighted_scores)


lr_search_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-05]
repeated_list = [elem for elem in lr_search_list for _ in range(11)]

import numpy as np
import pandas as pd

df = pd.DataFrame(np.c_[repeated_list, f1_weighted_scores], columns = ['lr', 'f1w'])

print(df)


print(df.loc[df['lr'] == 1e-5])