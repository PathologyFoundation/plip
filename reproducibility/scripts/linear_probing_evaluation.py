import sys
sys.path.append("../../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import logging
from reproducibility.embedders.factory import EmbedderFactory
from reproducibility.evaluation.linear_probing.linear_classifier import LinearProber
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from reproducibility.utils.results_handler import ResultsHandler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def config():
    load_dotenv("../config.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str, choices=['plip', 'clip', 'mudipath'])
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="Kather", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)

    ## Probe hparams
    parser.add_argument("--alpha", default=0.01, type=float)
    return parser.parse_args()


if __name__ == "__main__":

    args = config()

    np.random.seed(args.seed)

    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]

    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]
    
    print('Now working on:')
    print(f'    Dataset: {args.dataset}')
    print(f'    Model: {args.model_name}')
    print(f'    Backbone: {args.backbone}')

    train_dataset_name = args.dataset + "_train.csv"
    test_dataset_name = args.dataset + "_test.csv"

    train_dataset = pd.read_csv(os.path.join(data_folder, train_dataset_name))
    test_dataset = pd.read_csv(os.path.join(data_folder, test_dataset_name))

    embedder = EmbedderFactory().factory(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_x = embedder.image_embedder(train_dataset["image"].tolist(), additional_cache_name=train_dataset_name, device=device)
    test_x = embedder.image_embedder(test_dataset["image"].tolist(), additional_cache_name=test_dataset_name, device=device)

    prober = LinearProber(alpha=args.alpha, seed=args.seed)

    classifier, results = prober.train_and_test(train_x=train_x, train_y=train_dataset["label"].tolist(),
                                    test_x=test_x, test_y=test_dataset["label"].tolist())

    additional_parameters = {'dataset': args.dataset, 'seed': args.seed,
                             'model': args.model_name, 'backbone': args.backbone,
                             'alpha': args.alpha}

    rs = ResultsHandler(args.dataset, "linear_probing", additional_parameters)
    rs.add(results)

    ###############################################################
    # below are new codes
    ###############################################################
    opj = os.path.join
    savedir = opj(os.environ["PC_RESULTS_FOLDER"], args.dataset, args.model_name, 'seed=%d' % args.seed, 'alpha=' + str(args.alpha))
    os.makedirs(savedir, exist_ok=True)
    backbone = args.backbone

    if args.model_name == 'plip':
        backbone = os.path.basename(backbone)

    save_filename = opj(savedir, '%s.csv' % backbone)

    test_perf, train_perf = results
    train_perf = pd.DataFrame(train_perf, index=[0])
    test_perf = pd.DataFrame(test_perf, index=[1])
    perf = pd.concat([train_perf, test_perf], axis=0)
    perf.to_csv(save_filename)
