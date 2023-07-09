import sys
sys.path.append("../../")
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import pandas as pd
from dotenv import load_dotenv
import os
opj=os.path.join
import clip
import tqdm
import numpy as np
import random
import torch
from reproducibility.embedders.internal_datasets import CLIPCaptioningDataset, CLIPImageDataset
from reproducibility.embedders.transform import _train_transform
from torch.utils.data import DataLoader


def image_embedder(model, preprocess, list_of_images, device="cuda", num_workers=1, batch_size=32):
    print('Generating image embedding ...')
    train_dataset = CLIPImageDataset(list_of_images, preprocess)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    image_embeddings = []

    total = len(list_of_images) // batch_size
    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            image_embeddings.extend(model.encode_image(images).detach().cpu().numpy())

            pbar.update(1)
        pbar.close()

    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def text_embedder(model, list_of_labels, device="cuda", num_workers=1, batch_size=32):
    print('Generating text embedding ...')
    train_dataset = CLIPCaptioningDataset(list_of_labels)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    text_embeddings = []
    total = len(list_of_labels) // batch_size

    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for captions in dataloader:
            idx = clip.tokenize(captions, truncate=True).to(device)
            text_embeddings.extend(model.encode_text(idx).detach().cpu().numpy())

            pbar.update(1)

        pbar.close()

    text_embeddings = np.array(text_embeddings)
    return text_embeddings

    
def train_init(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def extract_embedding(args,
                        dataset_name='OpenPath',
                        df=None, # image-text pair in dataframe.
                        ):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_init(seed=args.random_seed)
    model, preprocess = clip.load(model_type,
                                    device=device,
                                    jit=False)  # Must set jit=False for training
    if dataset_name == 'OpenPath':
        print('Use train_preprocess, which first resize to 512, then crop into 224 by 224.')
        preprocess = _train_transform(first_resize = args.first_resize,
                                            n_px = args.pxsize
                                            )
    else:
        print('Use normal preprocess, which assume the input is with dimension 224 by 224.')
        preprocess = preprocess
        

    if args.model_name == 'plip':
        print(f'PLIP loading backbone: {args.backbone}')
        model.load_state_dict(torch.load(args.backbone))
    
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    
    # Get image embedding
    image_embeddings = image_embedder(model, preprocess, df['image'].values.astype(str), device=device, num_workers=args.num_workers, batch_size=args.batch_size)
    
    # Get text embedding
    text_embeddings = text_embedder(model, df['caption'].values.astype(str), device=device, num_workers=args.num_workers, batch_size=args.batch_size)
    return image_embeddings, text_embeddings

def config():
    load_dotenv("../config.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="plip", type=str, choices=['plip', 'clip'])
    parser.add_argument("--backbone", default='default', type=str)
    parser.add_argument("--dataset", default="OpenPath", type=str)
    parser.add_argument("--first_resize", default=512, type=int, help='This is image preprocessing transformation parameter.')
    parser.add_argument("--pxsize", default=224, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    args = config()
    
    savepath = opj('/oak/stanford/groups/jamesz/pathtweets/results/embeddings', args.dataset)
    os.makedirs(savepath, exist_ok=True)

    # Open README.md file in write mode
    with open(opj(savepath, '../..', 'README.md'), 'w') as readme_file:
        # Write your desired content
        readme_file.write('# Note\n\n')
        readme_file.write('The image and text embeddings are saved in numpy binary files (```.npy```). The order of the embeddings corresponds to their associated ```.csv``` files.\n')
        readme_file.write('Both unnormalized and normalized formats are available for all embeddings.\n\n')
        readme_file.write('The normalization process was performed using the equation: ```embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)```.\n')
        readme_file.write('The normalized embeddings were used for linear probing analysis.\n\n')


    data_folder = os.environ["PC_EVALUATION_DATA_ROOT_FOLDER"]
    if args.model_name == "plip" and args.backbone == "default":
        args.backbone = os.environ["PC_DEFAULT_BACKBONE"]
    
    model_type = os.environ["PC_CLIP_ARCH"]
    ##############################################################
    # Extract embedding for image and text of OpenPath.
    ##############################################################
    if args.dataset == 'OpenPath':
        dd = '/oak/stanford/groups/jamesz/pathtweets/v2/clean_dataset'
        df_T = pd.read_csv(opj(dd, 'T-noQ.csv'))
        df_T['source'] = 'Twitter'
        df_T['hashtag'] = [os.path.basename(os.path.dirname(v)) for v in df_T['image']]
        df_T['media ID'] = [os.path.basename(v).split('.')[0] for v in df_T['image']]
        #df_T = df_T.loc[['?' not in v for v in df_T['caption']]]
        df_R = pd.read_csv(opj(dd, 'R-noQ.csv'))
        df_R['hashtag'] = [os.path.basename(os.path.dirname(v)) for v in df_R['image']]
        df_R['media ID'] = [os.path.basename(v).split('.')[0] for v in df_R['image']]
        df_R['source'] = 'Twitter reply'
        #df_R = df_R.loc[['?' not in v for v in df_R['caption']]]
        df_L = pd.read_csv(opj(dd, 'L.csv')) # In LAION, we keep question marks.
        df_L['source'] = 'PathLAION'
        df_L['hashtag'] = '----'
        df_L['media ID'] = [os.path.basename(v).split('.')[0] for v in df_L['image']]
        df = pd.concat([df_T, df_R, df_L], axis=0)
        df['image'] = df['image'].str.replace('pathtweets_data_20230211', 'pathtweets_data_20230426')
        #df = df.drop_duplicates(subset=['image','caption'], keep='last')
        df_public = df[['source','hashtag','weblink','id','media ID','caption']].reset_index(drop=True)
        print(df_public)
        df_public.to_csv(opj(savepath, 'df_208414.csv'))
        #exit()
    elif args.dataset in ['Kather_train', 'Kather_test','PanNuke_train', 'PanNuke_test',
                    'DigestPath_train', 'DigestPath_test', 'WSSS4LUAD_binary_train', 'WSSS4LUAD_binary_test']:
        dd = '/oak/stanford/groups/jamesz/pathtweets/v2/evaluation_datasets/classification/'
        df = pd.read_csv(opj(dd, f'{args.dataset}.csv'), index_col=0)
        df = df.rename(columns={'text_style_4': 'caption'}) # Style 4 is the standard prompt. Style 0 is just label.
        df['image'] = df['image'].str.replace('pathtweets_data_20230211', 'pathtweets_data_20230426')
        print(df)
        if args.dataset.startswith('Kather'):
            df['filename'] = [os.path.basename(v) for v in df['image']]
            df_public = df[['filename','label', 'caption']].reset_index(drop=True)
        elif args.dataset.startswith('PanNuke'):
            df_public = df[['text_style_0', 'label', 'label_text', 'caption']].reset_index(drop=True)
            df_public = df_public.rename(columns={'text_style_0': 'tissue'})
            df_public['tissue'] = [v.replace('benign ', '').replace('malignant ', '') for v in df_public['tissue']]
            df_public['label'] = df_public['label'].astype(int)
        else:
            df_public = df[['label', 'label_text', 'caption']].reset_index(drop=True)
            df_public['label'] = df_public['label'].astype(int)

        df_public.to_csv(opj(savepath, f'{args.dataset}.csv'))
    #exit()
    image_embeddings, text_embeddings = extract_embedding(args, dataset_name=args.dataset, df=df)
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    np.save(opj(savepath, f'{args.dataset}_image_embeddings.npy'), image_embeddings)
    np.save(opj(savepath, f'{args.dataset}_text_embeddings.npy'), text_embeddings)

    np.save(opj(savepath, f'{args.dataset}_image_embeddings_normalized.npy'), image_embeddings_norm)
    np.save(opj(savepath, f'{args.dataset}_embeddings_normalized.npy'), text_embeddings_norm)
