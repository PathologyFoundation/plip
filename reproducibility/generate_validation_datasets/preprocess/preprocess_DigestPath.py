#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:30:19 2023

@author: zhihuang
"""

import os, platform, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import copy
import json
import argparse
opj = os.path.join

machine_name = list(platform.uname())[1]
print("Currently working on %s Machine" % machine_name)
workdir = '/{your working directory}'

def background_ratio(rgb,
                     threshold=200
                     ):
    bg_mask = (rgb[..., 0] >= threshold) & (rgb[..., 1] >= threshold) & (rgb[..., 2] >= threshold)
    bg_pixel_count = np.sum(bg_mask)
    ratio = bg_pixel_count / (rgb.shape[0]*rgb.shape[1])
    return ratio

def random_crop(img,
                msk=None,
                downsample=1,
                cropsize=224,
                crop_overlap=0.1,
                non_bg_threshold=0.5,
                ):
    
    new_size = (int(np.round(img.size[0]/downsample)), int(np.round(img.size[1]/downsample)))
    img = img.resize(new_size)
    
    
    if img.size[0] < cropsize or img.size[1] < cropsize:
        return None, None
    
    img_np = np.array(img)
    
    if msk is not None:
        msk = msk.resize(new_size, Image.Resampling.NEAREST)
        msk_np = np.array(msk)
        '''
        For some reason, the mask is not binary images.
        Probably due to jpg compression. So we need to manually binarize the image.
        So, whatever pixel value <= 10, we will consider it as 0.
        '''
        msk_np = (msk_np > 10).astype(int)
    
    x_list = np.arange(0, img_np.shape[0], cropsize*(1-crop_overlap)).astype(int)
    y_list = np.arange(0, img_np.shape[1], cropsize*(1-crop_overlap)).astype(int)
    
    
    imgs_all = []
    tumor_to_patch_ratio_all = []
    tumor_to_tissue_ratio_all = []
    tissue_ratio_all = []
    
    for x1 in x_list:
        for y1 in y_list:
            x2, y2 = x1+cropsize, y1+cropsize
            if x2 >= img_np.shape[0] or y2 >= img_np.shape[1]: continue
            
            img_patch_np = img_np[x1:x2, y1:y2, :]
            if msk is not None:
                msk_patch_np = msk_np[x1:x2, y1:y2]
            bg_ratio = background_ratio(img_patch_np)
            tissue_ratio = 1-bg_ratio
            if tissue_ratio < non_bg_threshold: continue
            
            if msk is not None:
                tumor_to_patch_ratio = np.sum(msk_patch_np > 0) / (msk_patch_np.shape[0] * msk_patch_np.shape[1])
                tumor_to_tissue_ratio = np.sum(msk_patch_np > 0) / (msk_patch_np.shape[0] * msk_patch_np.shape[1] * tissue_ratio)
            else:
                # negative images
                tumor_to_patch_ratio = 0
                tumor_to_tissue_ratio = 0
                
            imgs_all.append(img_patch_np)
            tissue_ratio_all.append(tissue_ratio)
            tumor_to_patch_ratio_all.append(tumor_to_patch_ratio)
            tumor_to_tissue_ratio_all.append(tumor_to_tissue_ratio)
            
    if len(imgs_all) == 0:
        return None, None
    
    imgs_all = np.stack(imgs_all)
    
    df_stat = pd.DataFrame(np.c_[tissue_ratio_all, tumor_to_patch_ratio_all, tumor_to_tissue_ratio_all],
                           columns=['tissue_ratio','tumor_to_patch_ratio','tumor_to_tissue_ratio'])
    df_stat['downsample'] = downsample
    df_stat['cropsize'] = cropsize
    df_stat['crop_overlap'] = crop_overlap
    df_stat['non_bg_threshold'] = non_bg_threshold
    return imgs_all, df_stat

def run_step_1_get_images(cropsize, crop_overlap, non_bg_threshold, downsample_list, resultdir):
    # =============================================================================
    #     Positives
    # =============================================================================
    print('Getting positives ...')
    list_of_imgs = np.sort([v for v in os.listdir(path2pos) if not v.endswith('_mask.jpg')])
    all_pos_imgs = None
    all_pos_stat = None
    for fname_img in tqdm(list_of_imgs):
        uniq_filename = fname_img.replace('.jpg', '')
        fname_mask = fname_img.replace('.jpg','_mask.jpg')
        img_source = Image.open(opj(path2pos, fname_img))
        msk_source = Image.open(opj(path2pos, fname_mask))
        this_file_imgs = None
        this_file_stat = None
        for downsample in downsample_list:
            imgs, stat = None, None
            imgs, stat = random_crop(img_source, msk_source, downsample, cropsize, crop_overlap, non_bg_threshold)
            if imgs is None: continue
            stat['filename'] = uniq_filename
            stat['downsample'] = downsample
            assert len(imgs) == len(stat)
            if this_file_imgs is None:
                this_file_imgs = imgs
                this_file_stat = stat
            else:
                this_file_imgs = np.concatenate([this_file_imgs, imgs], axis=0)
                this_file_stat = pd.concat([this_file_stat, stat], axis=0)
                assert len(this_file_imgs) == len(this_file_stat)
        if this_file_imgs is not None:
            if all_pos_imgs is None:
                all_pos_imgs = this_file_imgs
                all_pos_stat = this_file_stat
            else:
                all_pos_imgs = np.concatenate([all_pos_imgs, this_file_imgs], axis=0)
                all_pos_stat = pd.concat([all_pos_stat, this_file_stat], axis=0)
                assert len(all_pos_imgs) == len(all_pos_stat)
    all_pos_stat = all_pos_stat.reset_index(drop=True)
    all_pos_stat['from'] = 'tissue-train-pos-v1'
    
    np.save(opj(resultdir, 'imgs_from_pos_v1.npy'), all_pos_imgs)
    all_pos_stat.to_csv(opj(resultdir, 'stat_from_pos_v1.csv'))
    
    # =============================================================================
    #     Positives
    # =============================================================================
    print('Getting negatives ...')
    list_of_imgs = np.sort([v for v in os.listdir(path2neg) if not v.endswith('_mask.jpg')])
    all_neg_imgs = None
    all_neg_stat = None
    for fname_img in tqdm(list_of_imgs):
        uniq_filename = fname_img.replace('.jpg', '')
        img_source = Image.open(opj(path2neg, fname_img))
        msk_source = None
        this_file_imgs = None
        this_file_stat = None
        for downsample in downsample_list:
            imgs, stat = None, None
            imgs, stat = random_crop(img_source, msk_source, downsample, cropsize, crop_overlap, non_bg_threshold)
            if imgs is None: continue
            stat['filename'] = uniq_filename
            stat['downsample'] = downsample
            assert len(imgs) == len(stat)
            if this_file_imgs is None:
                this_file_imgs = imgs
                this_file_stat = stat
            else:
                this_file_imgs = np.concatenate([this_file_imgs, imgs], axis=0)
                this_file_stat = pd.concat([this_file_stat, stat], axis=0)
        if this_file_imgs is not None:
            if all_neg_imgs is None:
                all_neg_imgs = this_file_imgs
                all_neg_stat = this_file_stat
            else:
                all_neg_imgs = np.concatenate([all_neg_imgs, this_file_imgs], axis=0)
                all_neg_stat = pd.concat([all_neg_stat, this_file_stat], axis=0)
    all_neg_stat = all_neg_stat.reset_index(drop=True)
    all_neg_stat['from'] = 'tissue-train-neg'
    
    np.save(opj(resultdir, 'imgs_from_neg.npy'), all_neg_imgs)
    all_neg_stat.to_csv(opj(resultdir, 'stat_from_neg.csv'))
    
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default=1, type=int, choices=[1,2,3])
    return parser.parse_args()



if __name__=='__main__':
    
    args = parse_args()
    step = args.step
    
    dd = opj(workdir, 'data_validation', 'DigestPath2019', 'Colonoscopy_tissue_segment_dataset')
    
    path2neg = opj(dd, 'tissue-train-neg')
    path2pos = opj(dd, 'tissue-train-pos-v1')
    
    
    # =============================================================================
    #     Hyperparameters
    # =============================================================================
    cropsize = 224
    crop_overlap = 0.1
    non_bg_threshold = 0.5
    downsample_list = [2, 4, 8, 16, 32]
    
    tumor2patch_ratio_threshold = 0.5
    step_1_resultdir = opj(dd, 'processed',
                            'cropsize=%d_overlap=%.2f_nonbgthreshold=%.2f_downsamplelist=%s' % \
                            (cropsize, crop_overlap, non_bg_threshold, str(downsample_list)),
                            'step_1'
                        )
    step_2_resultdir = opj(dd, 'processed',
                    'cropsize=%d_overlap=%.2f_nonbgthreshold=%.2f_downsamplelist=%s' % \
                        (cropsize, crop_overlap, non_bg_threshold, str(downsample_list)),
                        'step_2_tumor2patch_ratio_threshold=%.2f' % tumor2patch_ratio_threshold,
                    )
    if step == 1:
        # =============================================================================
        #     Run step 1
        # =============================================================================
        
        os.makedirs(step_1_resultdir, exist_ok=True)
        run_step_1_get_images(cropsize, crop_overlap, non_bg_threshold, downsample_list, step_1_resultdir)
        
    elif step == 2:
    
        # =============================================================================
        #     Run step 2
        # =============================================================================
        '''
        You can clear memory right here.
        '''
        os.makedirs(step_2_resultdir, exist_ok=True)
        
        # step 2: prepare dataset
        imgs_from_neg = np.load(opj(step_1_resultdir, 'imgs_from_neg.npy'))
        stat_from_neg = pd.read_csv(opj(step_1_resultdir, 'stat_from_neg.csv'), index_col=0)
        
        imgs_from_pos_v1 = np.load(opj(step_1_resultdir, 'imgs_from_pos_v1.npy'))
        stat_from_pos_v1 = pd.read_csv(opj(step_1_resultdir, 'stat_from_pos_v1.csv'), index_col=0)
        pos_index = stat_from_pos_v1['tumor_to_patch_ratio'].values >= tumor2patch_ratio_threshold
        neg_index = stat_from_pos_v1['tumor_to_patch_ratio'].values == 0
        print('%d negative patches from pos_v1.' % np.sum(neg_index))
        print('%d positive patches (tumor2patch_ratio >= %.2f) from pos_v1.' % (np.sum(pos_index), tumor2patch_ratio_threshold))
        
        final_negative_images = np.concatenate([imgs_from_neg, imgs_from_pos_v1[neg_index, ...]], axis=0)
        final_negative_stats = pd.concat([stat_from_neg, stat_from_pos_v1.loc[neg_index]], axis=0).reset_index(drop=True)
        
        final_positive_images = imgs_from_pos_v1[pos_index, ...]
        final_positive_stats = stat_from_pos_v1.loc[pos_index]
        
        print('Finally, %d of negative images and %d of positive images' % (len(final_negative_stats), len(final_positive_stats)))
        #raise Exception()
        np.save(opj(step_2_resultdir, 'final_negative_images.npy'), final_negative_images)
        final_negative_stats.to_csv(opj(step_2_resultdir, 'final_negative_stats.csv'))
        
        np.save(opj(step_2_resultdir, 'final_positive_images.npy'), final_positive_images)
        final_positive_stats.to_csv(opj(step_2_resultdir, 'final_positive_stats.csv'))
    
    
    elif step == 3:
        # =============================================================================
        #     Run step 3: Convert npy files to png images.
        # =============================================================================
        '''
        You can clear memory right here.
        '''
        print('Run step 3: Convert npy files to png images.')
        
        imgs_from_neg = np.load(opj(step_2_resultdir, 'final_negative_images.npy'))
        imgs_from_pos_v1 = np.load(opj(step_2_resultdir, 'final_positive_images.npy'))

        stat_from_neg = pd.read_csv(opj(step_2_resultdir, 'final_negative_stats.csv'), index_col=0)
        stat_from_pos_v1 = pd.read_csv(opj(step_2_resultdir, 'final_positive_stats.csv'), index_col=0)

        # unstack negatives
        png_savedir_neg = opj(step_2_resultdir, 'images', 'negative')
        os.makedirs(png_savedir_neg, exist_ok=True)
        for i in tqdm(range(len(imgs_from_neg))):
            img_np = imgs_from_neg[i, ...]
            filename = stat_from_neg.iloc[i]['filename']
            downsample = stat_from_neg.iloc[i]['downsample']
            img = Image.fromarray(img_np)
            img.save(opj(png_savedir_neg, '%s_downsample=%d_%05d.png' % (filename, downsample, i)))

        # unstack positives
        png_savedir_pos = opj(step_2_resultdir, 'images', 'positive')
        os.makedirs(png_savedir_pos, exist_ok=True)
        for i in tqdm(range(len(imgs_from_pos_v1))):
            img_np = imgs_from_pos_v1[i, ...]
            filename = stat_from_pos_v1.iloc[i]['filename']
            downsample = stat_from_pos_v1.iloc[i]['downsample']
            img = Image.fromarray(img_np)
            img.save(opj(png_savedir_pos, '%s_downsample=%d_%05d.png' % (filename, downsample, i)))
        
    print('All done.')
