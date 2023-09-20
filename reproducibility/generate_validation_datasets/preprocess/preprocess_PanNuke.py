#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:35:00 2023

@author: zhihuang
"""

import os
opj = os.path.join
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

if __name__ == '__main__':
    dd = '/{your data path}/PanNuke'
    
    print("Load npy files ...")
    fold1_imgs = np.load(opj(dd, 'fold_1', 'Fold 1', 'images', 'fold1', 'images.npy')).astype(np.uint8)
    fold1_msks = np.load(opj(dd, 'fold_1', 'Fold 1', 'masks', 'fold1', 'masks.npy')).astype(np.uint8)
    fold1_typs = np.load(opj(dd, 'fold_1', 'Fold 1', 'images', 'fold1', 'types.npy'))
    
    fold2_imgs = np.load(opj(dd, 'fold_2', 'Fold 2', 'images', 'fold2', 'images.npy')).astype(np.uint8)
    fold2_msks = np.load(opj(dd, 'fold_2', 'Fold 2', 'masks', 'fold2', 'masks.npy')).astype(np.uint8)
    fold2_typs = np.load(opj(dd, 'fold_2', 'Fold 2', 'images', 'fold2', 'types.npy'))
    
    fold3_imgs = np.load(opj(dd, 'fold_3', 'Fold 3', 'images', 'fold3', 'images.npy')).astype(np.uint8)
    fold3_msks = np.load(opj(dd, 'fold_3', 'Fold 3', 'masks', 'fold3', 'masks.npy')).astype(np.uint8)
    fold3_typs = np.load(opj(dd, 'fold_3', 'Fold 3', 'images', 'fold3', 'types.npy'))
    
    imgs = np.concatenate([fold1_imgs, fold2_imgs, fold3_imgs], axis=0)
    msks = np.concatenate([fold1_msks, fold2_msks, fold3_msks], axis=0)
    typs = np.concatenate([fold1_typs, fold2_typs, fold3_typs], axis=0)
    
    print("Finished loading.")
    
    # Drop images that contains no cells
    idx = np.sum(msks[..., 0:5].reshape(len(msks), -1), axis=1) == 0
    print(f'{np.sum(idx)} images are purely background. Drop them.')
    imgs = imgs[~idx]
    msks = msks[~idx]
    typs = typs[~idx]
    
    print(f'Total images: {len(imgs)}')

    """
    ### Optional: Get the count of specific nuclei for each image
    ### 0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial, 6: Background
    n_nuclei = {}
    for i in range(6):
        n_nuclei[i] = np.sum(msks[..., i].reshape(msks.shape[0], -1), axis=1)
    """

    stat_ncells = pd.DataFrame(index=np.arange(len(imgs)), columns = np.arange(6))
    for i in tqdm(range(len(imgs))):
        for j in range(6):
            uniq_cells = len(np.unique(msks[..., j].reshape(msks.shape[0], -1)[i,:]))-1
            stat_ncells.loc[i, j] = uniq_cells
    total_cells = stat_ncells.sum(axis=1)
    
    print('Number of images contain neoplastic cells:', np.sum(stat_ncells[0] > 0), '/', len(imgs))
    print('Number of images contain inflammatory cells:', np.sum(stat_ncells[1] > 0), '/', len(imgs))
    print('Number of images contain epithelial cells:', np.sum(stat_ncells[4] > 0), '/', len(imgs))

    print('----------------------------------------------------------------------------------')
    print('Threshold to determine tumor: n_tumor >= 10, and at least 30% of cells are tumors.')
    tumor_idx = (stat_ncells[0] >= 10) & (stat_ncells[0]/total_cells > 0.3)
    print('Number of tumor images:', np.sum(tumor_idx), '/', len(imgs))
    
    print('----------------------------------------------------------------------------------')
    print('Threshold to determine benign: n_tumor == 0')
    benign_idx = (stat_ncells[0] == 0)
    print('Number of tumor images:', np.sum(benign_idx), '/', len(imgs))
    
    imgs_malignant, imgs_benign = imgs[tumor_idx, ...], imgs[benign_idx, ...]
    typs_malignant, typs_benign = typs[tumor_idx, ...], typs[benign_idx, ...]
    
    
    uniq_types = np.unique(typs)
    for ttype in uniq_types:
        print(f'{ttype}\t maligant: {np.sum(typs_malignant == ttype)}\t benign: {np.sum(typs_benign == ttype)}')
    
    savedir = opj(dd, 'processed_threshold=10_0.3', 'images')
    os.makedirs(savedir, exist_ok=True)
    
    df = pd.DataFrame()
    
    for i in tqdm(range(len(imgs_malignant))):
        img = Image.fromarray(imgs_malignant[i, ...])
        tissue = str(typs_malignant[i, ...]).lower().replace('_', ' ')
        fname = '%s_malignant_%04d.png' % (tissue, i)
        img.save(opj(savedir, fname))
        
        caption = 'An H&E image of malignant %s tissue.' % tissue
        row = pd.DataFrame({'image': opj(savedir, fname),
                            'caption': caption}, index=[i])
        df = pd.concat([df, row], axis=0)
    
    for i in tqdm(range(len(imgs_benign))):
        img = Image.fromarray(imgs_benign[i, ...])
        tissue = str(typs_benign[i, ...]).lower().replace('_', ' ')
        fname = '%s_benign_%04d.png' % (tissue, i)
        img.save(opj(savedir, fname))
        
        caption = 'An H&E image of benign %s tissue.' % tissue
        row = pd.DataFrame({'image': opj(savedir, fname),
                            'caption': caption}, index=[i])
        df = pd.concat([df, row], axis=0)
    
    df.to_csv(opj(dd, 'processed_threshold=10_0.3', 'PanNuke_all_binary.csv'))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
