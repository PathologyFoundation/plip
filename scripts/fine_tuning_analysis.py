import sys
sys.path.append("../")
import argparse
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import os
opj = os.path.join
import numpy as np
from utils.results_handler import ResultsHandler




def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage_of_training_data", default=1.0, type=float,
                        help="""The ratio of the training data (range 0.0 - 1.0).
                                If value = 1, use all training data to fine-tune.
                                If value = 0.2, use 20%% of the training data to fine-tune.""")
    parser.add_argument("--valid_ratio", default=0.3, type=float,
                        help="""The ratio of the validation set that came from training data.
                                If sub-sampling was performed on the training data, the validation set
                                is generated using the sub-sampled portion.""")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--optimizer", default='AdamW', type=str)
    parser.add_argument("--save_directory", default='/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning')
    parser.add_argument("--random_seed", default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":

    args = config()

    datasets = ['Kather', 'PanNuke', 'DigestPath', 'WSSS4LUAD_binary']
    train_ratios = [0.01, 0.1, 0.5, 1]
    model_list = ['clip','plip','MuDiPath','EfficientNet_b0','EfficientNet_b3','EfficientNet_b7','resnet50','resnet101','vit_b_32','vit_b_16']

    ###############################################################
    # Step 1. Get all results
    ###############################################################
    multicol = pd.MultiIndex.from_product([datasets, train_ratios], names=['dataset','train_ratio'])
    perf_df = pd.DataFrame(index=model_list, columns=multicol)

    for dataset in datasets:
        for model in model_list:
            for train_ratio in train_ratios:
                if model == 'plip':
                    savesubdir = f'PLIP_btch={args.batch_size}_wd={args.weight_decay}_nepochs={args.epochs}_validratio={args.valid_ratio}_optimizer={args.optimizer}'
                else:
                    savesubdir = f'{model}'

                # Get result folder
                result_folder = None
                result_parent_folder = opj(args.save_directory, dataset, f'train_ratio={float(train_ratio)}', savesubdir)
                if not os.path.exists(result_parent_folder): continue
                result_seed_dirs = os.listdir(result_parent_folder)
                result_folder = [opj(result_parent_folder, v) for v in result_seed_dirs if int(v.split('random_seed=')[1].split('_')[0]) == args.random_seed]
                result_folder = np.sort(result_folder)
                if len(result_folder) == 1:
                    result_folder = result_folder[0]
                elif len(result_folder) > 1:
                    result_folder = result_folder[-1]
                else:
                    #raise Exception('Parent folder exists but empty inside.')
                    continue

                # Get test performance
                candidate_filenames = np.array(os.listdir(result_folder)).astype(str)
                test_csv_filename = None
                test_csv_filename = [opj(result_folder, v) for v in candidate_filenames if v.startswith('performance_test_best_lr')]
                if len(test_csv_filename) == 0:
                    continue
                elif len(test_csv_filename) == 1:
                    test_csv_filename = test_csv_filename[0]
                else:
                    raise Exception('This does not make sense.')
                test_performance = pd.read_csv(test_csv_filename, sep='\t', index_col=0)

                #print(test_performance)

                f1_w = test_performance['f1_weighted'].values[-1]
                perf_df.loc[model, (dataset, train_ratio)] = f1_w

    print('---------------------------------------------------------')
    print(perf_df)
    










