import sys
sys.path.append("../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import numpy as np
import logging
from embedders.factory import EmbedderFactory
from evaluation.zero_shot.zero_shot import ZeroShotClassifier
import pandas as pd
from dotenv import load_dotenv
import os
from utils.results_handler import ResultsHandler
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from PIL import Image
def config():
    load_dotenv("../config.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str)
    parser.add_argument("--caption_column", default="text_style_4", type=str)
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="kather", type=str)
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

    test_dataset_name = args.dataset + "_test.csv"

    test_dataset = pd.read_csv(os.path.join(data_folder, test_dataset_name))
    print("Pixels", np.array(Image.open(test_dataset["image"].tolist()[0])))
    embedder = EmbedderFactory().factory(args)

    test_x = embedder.image_embedder(test_dataset["image"].tolist(),
                                     additional_cache_name=test_dataset_name, batch_size=124)
    print(test_x[0][0:10])
    exit()
    labels = test_dataset["label"].unique().tolist()

    # embeddings are generated using the selected caption, not the labels
    test_y = embedder.text_embedder(test_dataset[args.caption_column].unique().tolist(),
                                    additional_cache_name=test_dataset_name, batch_size=512)

    prober = ZeroShotClassifier()

    results = prober.zero_shot_classification(test_x, test_y,
                                              unique_labels=labels, target_labels=test_dataset["label"].tolist())

    additional_parameters = {'dataset': args.dataset, 'seed': args.seed,
                             'model': args.model_name, 'backbone': args.backbone}

    rs = ResultsHandler(args.dataset, "zero_shot", additional_parameters)
    rs.add(results)

