import sys
sys.path.append("../../")
import argparse
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import os
opj = os.path.join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import copy

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

    return parser.parse_args()


if __name__ == "__main__":

    args = config()

    datasets = ['Kather_train', 'PanNuke', 'DigestPath', 'WSSS4LUAD_binary']
    train_ratios = [0.01, 0.05, 0.1, 0.5, 1]
    model_list = ['plip','vit_b_32']
    ###############################################################
    # Step 1. Get all results
    ###############################################################
    random_seeds = np.arange(10)
    multicol = pd.MultiIndex.from_product([datasets, train_ratios, random_seeds], names=['dataset','train_ratio','random_seed'])
    perf_df = pd.DataFrame(index=model_list, columns=multicol)

    for dataset in datasets:
        for model in model_list:
            for train_ratio in train_ratios:
                for random_seed in random_seeds:
                    if model == 'plip':
                        savesubdir = f'PLIP_btch={args.batch_size}_wd={args.weight_decay}_nepochs={args.epochs}_validratio={args.valid_ratio}_optimizer={args.optimizer}'
                    else:
                        savesubdir = f'{model}'

                    # Get result folder
                    result_folder = None
                    result_parent_folder = opj(args.save_directory, dataset, f'train_ratio={float(train_ratio)}', savesubdir)
                    if not os.path.exists(result_parent_folder): continue
                    result_seed_dirs = os.listdir(result_parent_folder)
                    result_folder = [opj(result_parent_folder, v) for v in result_seed_dirs if int(v.split('random_seed=')[1].split('_')[0]) == random_seed]
                    result_folder = np.sort(result_folder)
                    if len(result_folder) == 1:
                        result_folder = result_folder[0]
                    elif len(result_folder) > 1:
                        #result_folder = result_folder[-1]
                        # find out which folder contains the result.
                        result_found = False
                        for rs in result_folder:
                            matching_files = glob.glob(opj(rs, 'performance_test_*.tsv'))
                            if len(matching_files):
                                result_folder = rs
                                result_found = True
                                break
                        if not result_found:
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
                    perf_df.loc[model, (dataset, train_ratio, random_seed)] = f1_w

    print('---------------------------------------------------------')
    #print(perf_df.astype(float).round(decimals=3).T)

    for dataset in datasets:
        temp = perf_df.loc[:, perf_df.columns.get_level_values('dataset')==dataset]
        print(f'Dataset: {dataset}')
        print(temp.astype(float).round(decimals=3).T)

    


    #######################################
    # Aggregate performance across four datasets and get mean
    #######################################
    multicol = pd.MultiIndex.from_product([datasets, train_ratios], names=['dataset','train_ratio'])
    perf_df_mean = pd.DataFrame(index=perf_df.index, columns=multicol)
    for model in perf_df.index:
        for dataset in datasets:
            for train_ratio in train_ratios:
                val = perf_df.loc[model, (perf_df.columns.get_level_values('dataset')== dataset) & (perf_df.columns.get_level_values('train_ratio')== train_ratio)]

                if np.isnan(val.values.astype(float)).all():
                    continue
                mean = np.nanmean(val.values)
                std = np.nanstd(val.values)
                perf_df_mean.loc[model, (dataset, train_ratio)] = f'{mean:.3f}±{std:.3f}'
    print('---------------------------------------------------------')
    print('Mean performance by averaging datasets')
    print(perf_df_mean)

    ###################################################################
    # Now start plotting
    ###################################################################
    savedir = '/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning/__figures'
    os.makedirs(savedir, exist_ok=True)

    # Move the second level of columns to the second index level
    temp_df = copy.deepcopy(perf_df_mean)
    temp_df.columns = temp_df.columns.set_levels(temp_df.columns.levels[1], level=1)
    temp_df = temp_df.stack(level=1)
    temp_df.reset_index(level=[0, 1], drop=False, inplace=True)
    temp_df.sort_values(by='train_ratio', inplace=True)
    temp_df.to_csv(opj(savedir, 'perf_mean.csv'))

    number_of_train_data = {'Kather_train': 90000, 'PanNuke': 4346, 'DigestPath': 43899, 'WSSS4LUAD_binary': 7063}


    axis_label = ['a','b','c','d']
    fig, ax = plt.subplots(1, len(datasets), figsize=(16,4), sharey=False)
    for i, dataset in enumerate(datasets):
        this_perf_df = perf_df.loc[:, perf_df.columns.get_level_values('dataset')==dataset]
        # Rename the index
        this_perf_df.rename(index={'vit_b_32': 'ViT-B/32',
                                    'plip': 'PLIP image encoder'}, inplace=True)
        this_perf_df = this_perf_df.stack()
        #print(this_perf_df)
        # Set the x-axis label and tick labels
        ax[i].set_xlabel('Proportion of training data used')
        xticks = this_perf_df.columns.get_level_values('train_ratio')
        n_datas = [int(np.round(v*number_of_train_data[dataset])) for v in this_perf_df.columns.get_level_values('train_ratio')]
        xticks = ['%d%%\n(N=%d)' % (v*100, n_data) for v, n_data in zip(xticks, n_datas)]
        ax[i].set_xticks(range(len(this_perf_df.columns)), xticks, rotation=0)
        ax[i].text(-0.15, 1.05, f'{axis_label[i]}', transform=ax[i].transAxes, fontweight='bold', fontsize=16)

        this_perf_df.columns = np.arange(len(this_perf_df.columns))


        sns.lineplot(data=this_perf_df.T,
                    palette=sns.color_palette("muted", len(this_perf_df)),
                    marker='o',
                    errorbar=('ci', 95),
                    #errorbar=('sd'),
                    ax=ax[i]
                    )
        # Set the y-axis label
        ax[i].set_ylabel('Weighted F1')
        # Set the y-axis to display values with two digits
        ax[i].yaxis.set_major_formatter('{x:.2f}')

        # Set the title
        if dataset == 'Kather_train':
            dataset = 'Kather colon (training split)'
        elif dataset == 'WSSS4LUAD_binary':
            dataset = 'WSSS4LUAD'
        ax[i].set_title(dataset)
    fig.tight_layout()
    fig.savefig(opj(savedir,'performance.png'), dpi=300)
    fig.savefig(opj(savedir,'performance.pdf'))






