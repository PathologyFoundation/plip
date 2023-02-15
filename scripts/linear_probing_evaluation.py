import sys
sys.path.append("../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from embedders.factory import EmbedderFactory
from evaluation.linear_probing.linear_classifier import LinearProber
import pandas as pd
from dotenv import load_dotenv
import os

def config():
    load_dotenv("../config.env")

    DEFAULT_BACKBONE = "/oak/stanford/groups/jamesz/fede/medical_clip/novel_models_for_path/epoch_3_2023-01-30 14:57:58.402744_prime_bracket_4833.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, type=str)
    parser.add_argument("--dataset", default="kather", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)

    ## Probe hparams
    parser.add_argument("--alpha", default=0.01, type=float)
    return parser.parse_args()


if __name__ == "__main__":

    args = config()
    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]

    train_dataset_name = args.dataset + "_train.csv"
    test_dataset_name = args.dataset + "_test.csv"

    train_dataset = pd.read_csv(os.path.join(data_folder, train_dataset_name))
    test_dataset = pd.read_csv(os.path.join(data_folder, test_dataset_name))

    embedder = EmbedderFactory().factory(args.model_name, args.backbone)

    train_x = embedder.image_embedder(train_dataset["images"].tolist(), additional_cache_name=train_dataset_name)
    test_x = embedder.image_embedder(test_dataset["images"].tolist(), additional_cache_name=test_dataset_name)

    prober = LinearProber(alpha=args.alpha)

    results = prober.train_and_test(train_x, train_y=train_dataset["labels"].tolist(),
                                    test_x=test_x, test_y=test_dataset["labels"].tolist())

