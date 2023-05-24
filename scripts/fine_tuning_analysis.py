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
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.insert(0, '/oak/stanford/groups/jamesz/pathtweets/ML_scripts/utils')
import install_font


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
    train_ratios = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    train_ratios = [0.01, 0.1, 0.5, 1]
    model_list = ['clip','plip','MuDiPath','EfficientNet_b0','EfficientNet_b3','EfficientNet_b7','resnet50','resnet101','vit_b_32','vit_b_16']
    #model_list = ['plip','clip','MuDiPath','EfficientNet_b7','vit_b_32']
    #model_list = ['plip','clip','MuDiPath','EfficientNet_b0','EfficientNet_b7','vit_b_32']
    #model_list = ['plip','MuDiPath','EfficientNet_b0','EfficientNet_b7','vit_b_32']
    model_list = ['plip','EfficientNet_b7','vit_b_32']

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
    print(perf_df.astype(float).round(decimals=3))
    

    ###################################################################
    # Now start plotting
    ###################################################################
    savedir = '/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning/__figures'
    os.makedirs(savedir, exist_ok=True)

    fig, ax = plt.subplots(1, len(datasets), figsize=(16,4), sharey=False)
    for i, dataset in enumerate(datasets):
        this_perf_df = perf_df.loc[:, dataset]
        # Rename the index
        this_perf_df.rename(index={'EfficientNet_b7': 'EfficientNet',
                                    'vit_b_32': 'ViT-B/32',
                                    'plip': 'PLIP image encoder'}, inplace=True)
        # Set the x-axis label and tick labels
        ax[i].set_xlabel('Proportion of training data used')
        ax[i].set_xticks(range(len(this_perf_df.columns)), this_perf_df.columns, rotation=0)

        this_perf_df.columns = np.arange(len(this_perf_df.columns))
        sns.lineplot(data=this_perf_df.T,
                    palette=sns.color_palette("muted", len(this_perf_df)),
                    marker='o',
                    ax=ax[i]
                    )
        # Set the y-axis label
        ax[i].set_ylabel('Weighted F1')
        # Set the y-axis to display values with two digits
        ax[i].yaxis.set_major_formatter('{x:.2f}')

        # Set the title
        if dataset == "Kather":
            dataset = 'Kather colon'
        elif dataset == 'WSSS4LUAD_binary':
            dataset = 'WSSS4LUAD'
        ax[i].set_title(dataset)
    fig.tight_layout()
    fig.savefig(opj(savedir,'performance.png'), dpi=300)
    fig.savefig(opj(savedir,'performance.pdf'))






