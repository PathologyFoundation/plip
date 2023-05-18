import sys
sys.path.append("../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import logging
from embedders.factory import EmbedderFactory
from evaluation.fine_tuning.fine_tuning_classifier import FineTuner
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from utils.results_handler import ResultsHandler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def config():
    load_dotenv("../config.env")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str, choices=['plip'])
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="Kather", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    args = config()

    np.random.seed(args.seed)

    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]

    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]
    