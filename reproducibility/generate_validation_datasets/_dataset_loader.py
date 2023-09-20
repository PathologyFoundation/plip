import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
opj=os.path.join
ImageFile.LOAD_TRUNCATED_IMAGES = False

def process_Kather_csv(root_dir, seed=None):

    subtype_dict = {'ADI': 'adipose tissue',
                    'BACK': 'background',
                    'DEB': 'debris',
                    'LYM': 'lymphocytes',
                    'MUC': 'mucus',
                    'MUS': 'smooth muscle',
                    'NORM': 'normal colon mucosa',
                    'STR': 'cancer-associated stroma',
                    'TUM': 'colorectal adenocarcinoma epithelium'
                    }

    def prompt_engineering(text=''):
        prompt = 'An H&E image patch of [].'.replace('[]', text)
        return prompt

    KATHER100K_CSV = opj(root_dir, "data_validation", "Kather_100K_Colon", "image_fullpath_text_pair_100K.csv")
    KATHER7K_CSV = opj(root_dir, "data_validation", "Kather_100K_Colon", "image_fullpath_text_pair_7K_validation.csv")

    def process_csv(path2csv, root_dir, subtype_dict):
        df = pd.read_csv(path2csv)
        df = df[["image_fullpath", "label"]]
        df.columns = ['image', 'label']
        df['image'] = [root_dir + '/' + v.split('pathtweets/')[1] for v in df['image']]
        df['label_text'] = [subtype_dict[v] for v in df['label']]
        style=4
        df_all = pd.DataFrame()
        for subtype in subtype_dict.keys():
            df_subtype = df.loc[df['label'] == subtype]
            df_subtype['text_style_%d' % style] = prompt_engineering(subtype_dict[subtype])
            df_all = pd.concat([df_all, df_subtype], axis=0)
        df_all = df_all.reset_index(drop=True)
        return df_all
    
    train = process_csv(KATHER100K_CSV, root_dir, subtype_dict)
    test = process_csv(KATHER7K_CSV, root_dir, subtype_dict)

    return train, test



def process_WSSS4LUAD_binary(root_dir, seed, train_ratio):

    def prompt_engineering(text=''):
        prompt = 'An H&E image patch of [] tissue.'.replace('[]', text)
        return prompt

    path2data = opj(root_dir, 'data_validation', 'WSSS4LUAD', '1.training', '1.training')
    
    lbl2text = {0: 'normal', 1: 'tumor'}
    df = pd.DataFrame()
    for file in tqdm(os.listdir(path2data)):
        image_fullpath = opj(path2data, file)
        class_ = np.array(file.split('[')[1].split(']')[0].split(', ')).astype(int) # Multi-class labels: [Tumor, Stroma, Normal]
        if class_[0] == 1:
            lbl = 1 # has tumor
        else:
            lbl = 0 # no tumor
        try:
            Image.open(image_fullpath)
        except:
            print('Image %s cannot open. skip loading.' % file)
            continue
        row = pd.DataFrame({'image': image_fullpath, 
                           'label': lbl,
                           'label_text': lbl2text[lbl], 
                           }, index=[0])
        df = pd.concat([df, row], axis=0)
    df = df.reset_index(drop=True)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # randomly split data into training and testing.
    df_train = df.iloc[:int(len(df)*train_ratio),:].reset_index(drop=True)
    df_test = df.iloc[int(len(df)*train_ratio):,:].reset_index(drop=True)


    def process_csv(df_in):
        label_texts = ['tumor', 'normal']
        df_all = pd.DataFrame()
        for subtype in label_texts:
            df_subtype = df_in.loc[df_in['label_text'] == subtype]
            style = 4
            df_subtype['text_style_%d' % style] = prompt_engineering(subtype)
            df_all = pd.concat([df_all, df_subtype], axis=0)
        df_all = df_all.reset_index(drop=True)
        return df_all
    
    train = process_csv(df_train)
    test = process_csv(df_test)

    return train, test


def process_DigestPath(root_dir, seed=None, train_ratio=None):
    
    def prompt_engineering(text=''):
        prompt = 'An H&E image patch of [] tissue.'.replace('[]', text)
        return prompt

    dd = opj(root_dir, 'data_validation', 'DigestPath2019', 'Colonoscopy_tissue_segment_dataset',
             'processed', 'cropsize=224_overlap=0.10_nonbgthreshold=0.50_downsamplelist=[2, 4, 8, 16, 32]',
            'step_2_tumor2patch_ratio_threshold=0.30')

    final_negative_stats = pd.read_csv(opj(dd, 'final_negative_stats.csv'), index_col=0)
    final_positive_stats = pd.read_csv(opj(dd, 'final_positive_stats.csv'), index_col=0)
    n_neg = len(final_negative_stats)
    n_pos = len(final_positive_stats)

    final_negative_stats['filename'] = ["%05d" % v for v in final_negative_stats.index]
    final_positive_stats['filename'] = ["%05d" % v for v in final_positive_stats.index]

    df_neg = pd.DataFrame(index=range(n_neg), columns=['label'])
    df_pos = pd.DataFrame(index=range(n_pos), columns=['label'])

    df_neg['image'] = [opj(dd, 'images', 'negative', '%05d.png' % (i)) for i, (filename, downsample) in enumerate(zip(final_negative_stats['filename'], final_negative_stats['downsample']))]
    df_pos['image'] = [opj(dd, 'images', 'positive', '%05d.png' % (i)) for i, (filename, downsample) in enumerate(zip(final_positive_stats['filename'], final_positive_stats['downsample']))]
    df_neg['label'] = 0
    df_neg['label_text'] = 'benign'
    df_pos['label'] = 1
    df_pos['label_text'] = 'malignant'
    df = pd.concat([df_neg, df_pos], axis=0).reset_index(drop=True)
    df = df[['image','label','label_text']]
    
    uniq_sample_neg = final_negative_stats['filename'].unique()
    uniq_sample_pos = final_positive_stats['filename'].unique()
    np.random.seed(seed)
    np.random.shuffle(uniq_sample_neg)
    np.random.shuffle(uniq_sample_pos)
    
    train_samples = list(uniq_sample_neg[:int(len(uniq_sample_neg)*train_ratio)]) + \
                    list(uniq_sample_pos[:int(len(uniq_sample_pos)*train_ratio)])
    
    test_samples = list(uniq_sample_neg[int(len(uniq_sample_neg)*train_ratio):]) + \
                    list(uniq_sample_pos[int(len(uniq_sample_pos)*train_ratio):])
    
    print('Splitting training and testing data, balanced for neg and pos subgroups.')
    print(f'Train samples: {len(train_samples)}, test samples: {len(test_samples)}.')
    # make sure they are mutually exclusive, no data leaking
    #assert len(np.intersect1d(train_samples, test_samples)) == 0
    
    train_idx = np.isin([os.path.basename(v).split('_downsample')[0] for v in df['image']], train_samples)
    test_idx = np.isin([os.path.basename(v).split('_downsample')[0] for v in df['image']], test_samples)

    df_train = df.loc[train_idx,:].reset_index(drop=True)
    df_test = df.loc[test_idx,:].reset_index(drop=True)

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # randomly split data into training and testing.
    df_train = df.iloc[:int(len(df)*train_ratio),:].reset_index(drop=True)
    df_test = df.iloc[int(len(df)*train_ratio):,:].reset_index(drop=True)
    
    def process_csv(df_in):
        label_texts = ['benign', 'malignant']
        df_all = pd.DataFrame()
        for subtype in label_texts:
            df_subtype = df_in.loc[df_in['label_text'] == subtype]
            style = 4
            df_subtype['text_style_%d' % style] = prompt_engineering(subtype)
            df_all = pd.concat([df_all, df_subtype], axis=0)
        df_all = df_all.reset_index(drop=True)
        return df_all
    
    train = process_csv(df_train)
    test = process_csv(df_test)

    return train, test
    


def process_PanNuke(root_dir, seed=None, train_ratio=None):
    df = pd.read_csv('{Path to PanNuke dataset}/processed_threshold=10_0.3/PanNuke_all_binary.csv',index_col=0)
    df = df.reset_index(drop=True)
    for i in df.index:
        caption = df.loc[i, 'caption']
        if 'malignant' in caption:
            tissue = caption.split('malignant ')[1].split(' tissue')[0]
            df.loc[i, 'tissue'] = tissue
            df.loc[i, 'label'] = 1
            df.loc[i, 'label_text'] = 'malignant'
            df.loc[i, 'label_tissue'] = 'malignant %s' % tissue
            df.loc[i, 'caption_no_tissue'] = caption.replace(tissue + ' ', '')
        elif 'benign' in caption:
            tissue = caption.split('benign ')[1].split(' tissue')[0]
            df.loc[i, 'tissue'] = tissue
            df.loc[i, 'label'] = 0
            df.loc[i, 'label_text'] = 'benign'
            df.loc[i, 'label_tissue'] = 'benign %s' % tissue
            df.loc[i, 'caption_no_tissue'] = caption.replace(tissue + ' ', '')
        else:
            print(caption)
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    uniq_tissuetypes = df['tissue'].unique()
    

    # equally split dataset into train and test for each cancer subtype and each label
    train = pd.DataFrame()
    test = pd.DataFrame()
    for tissue in uniq_tissuetypes:
        for label_text in ['benign', 'malignant']:
            df_subset = df.loc[(df['tissue'] == tissue) & (df['label_text'] == label_text)]

            # shuffle data
            df_subset = df_subset.sample(frac=1, random_state=seed).reset_index(drop=True)
            # randomly split data into training and testing.
            df_subset_train = df_subset.iloc[:int(len(df_subset)*train_ratio),:].reset_index(drop=True)
            df_subset_test = df_subset.iloc[int(len(df_subset)*train_ratio):,:].reset_index(drop=True)

            train = pd.concat([train, df_subset_train], axis=0)
            test = pd.concat([test, df_subset_test], axis=0)
    
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    train = train[['image', 'label', 'label_text', 'label_tissue', 'caption', 'caption_no_tissue']]
    train.columns = ['image', 'label', 'label_text', 'text_style_0', 'text_style_1', 'text_style_4']

    test = test[['image', 'label', 'label_text', 'label_tissue', 'caption', 'caption_no_tissue']]
    test.columns = ['image', 'label', 'label_text', 'text_style_0', 'text_style_1', 'text_style_4']

    return train, test



def process_KIMIA_Path24(root_dir, seed=None):

    test_folder = opj(root_dir, 'data_validation', 'KIMIA_Path24C','Test-patches')
    data = []
    for label in os.listdir(test_folder):
        for jpg in os.listdir(opj(test_folder, label)):
            jpg_fullpath = opj(test_folder, label, jpg)
            data += [(jpg_fullpath, label)]
    test = pd.DataFrame(data, columns = ['image', 'label'])
    
    return test

