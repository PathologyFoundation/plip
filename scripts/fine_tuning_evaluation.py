import sys
sys.path.append("../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import logging
import time
from embedders.factory import EmbedderFactory
from evaluation.fine_tuning.fine_tuning_classifier import FineTuner
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
opj = os.path.join
import numpy as np
from utils.results_handler import ResultsHandler
from fine_tuning.clip import CLIPTuner
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def config():
    load_dotenv("../config.env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str, choices=['plip', 'clip', 'mudipath'])
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="Kather", type=str)
    parser.add_argument("--percentage_of_training_data", default=1.0, type=float,
                        help="""The ratio of the training data (range 0.0 - 1.0).
                        If value = 1, use all training data to fine-tune.
                        If value = 0.2, use 20%% of the training data to fine-tune.""")
    parser.add_argument("--seed", default=1, type=int)

    ## Fine-tuning hparams
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--first_resize", default=512, type=int, help='This is image preprocessing transformation parameter.')
    parser.add_argument("--pxsize", default=224, type=int)
    parser.add_argument("--optimizer", default='AdamW', type=str)
    parser.add_argument("--valid_ratio", default=0.10, type=float)
    parser.add_argument("--evaluation-steps", default=200, type=int)
    parser.add_argument("--save_directory", default='/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning')
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--comet-tracking", default=False)
    parser.add_argument("--comet_tags", nargs="*")
    parser.add_argument("--random_seed", default=0, type=int)

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

    #TODO This is currently hard-coded. May need to refactorize.
    train_dataset = train_dataset[['image', 'text_style_4']].rename(columns={'text_style_4': 'caption'}) # this is hard-coded
    test_dataset = test_dataset[['image', 'text_style_4']].rename(columns={'text_style_4': 'caption'}) # this is hard-coded
    train_dataset['image'] = train_dataset['image'].replace('pathtweets_data_20230211', 'pathtweets_data_20230426')
    test_dataset['image'] = test_dataset['image'].replace('pathtweets_data_20230211', 'pathtweets_data_20230426')

    embedder = EmbedderFactory().factory(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(embedder.model)




    train, valid = train_test_split(train_dataset, test_size=args.valid_ratio,
                                            random_state=args.random_seed,
                                            shuffle=True)
    
    TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
    savesubdir = f'{TIMESTRING}_data={args.dataset}_btch={args.batch_size}_lr={args.learning_rate}_'+\
                    f'wd={args.weight_decay}_firstresize={args.first_resize}_pxsize={args.pxsize}_nepochs={args.epochs}_'+\
                    f'validratio={args.valid_ratio}_optimizer={args.optimizer}'
                    
    args.save_directory = opj(args.save_directory, savesubdir)
    os.makedirs(args.save_directory, exist_ok=True)
    
    args_df = pd.DataFrame(vars(args),index=['Value']).T
    args_df.to_csv(opj(args.save_directory, 'arguments.csv'))
    
    print('------------------------------')
    print(args_df)
    print('------------------------------')

    
    logging.basicConfig(filename=opj(args.save_directory, '_training.log'),
                        format='%(asctime)s.%(msecs)03d *** %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )

    args.comet_tracking = None
    if args.model_name in ["clip", "plip"]:
        cpt = CLIPTuner(args=args,
                        logging=logging,
                        model_type=os.environ["PC_CLIP_ARCH"],
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        px_size=args.pxsize,
                        comet_tracking=args.comet_tracking,
                        comet_tags=args.comet_tags
                        )

        model_name = cpt.tuner(train, valid, save_directory=args.save_directory, batch_size=args.batch_size,
                epochs=args.epochs, evaluation_steps=args.evaluation_steps, num_workers=args.num_workers)

