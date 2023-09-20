import sys, os, platform, copy
from _dataset_loader import (process_Kather_csv,
                            process_PanNuke,
                            process_DigestPath,
                            process_WSSS4LUAD_binary,
                            process_KIMIA_Path24
                            )
import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import numpy as np
import multiprocess as mp
from tqdm import tqdm
from PIL import Image, ImageFile
opj=os.path.join
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


def resizeimg(fp, this_savedir):
    pbar.update(mp.cpu_count())
    newsize = 224
    img = Image.open(fp)
    filename = os.path.basename(fp)
    if img.size[0] != img.size[1]:
        width, height = img.size
        min_dimension = min(width, height) # Determine the smallest dimension
        scale_factor = newsize / min_dimension # Calculate the scale factor needed to make the smallest dimension 224
        # Calculate the new size of the image
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height)) # Resize the image using the calculated size
        # center crop
        left = (width - newsize) / 2 # Calculate the coordinates to crop the center of the image
        top = (height - newsize) / 2
        right = left + newsize
        bottom = top + newsize
        img_resize = img.crop((left, top, right, bottom)) # Crop the image using the calculated coordinates
    else:
        img_resize = img.resize((newsize, newsize))
    new_savename = opj(this_savedir, filename)
    img_resize.save(new_savename)
    return new_savename

if __name__ == '__main__':

    seed=1
    train_ratio=0.7

    img_savedir = '/{your data path}/data_validation_images_resize=224'
    savedir = opj('/{your data path}/evaluation_datasets', 'trainratio=%.2f_size=224' % (train_ratio))
    os.makedirs(img_savedir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    root_dir = '/{root directory}'

    #############################################
    #    Kather (9 classes)
    #############################################

    print('Processing Kather dataset ...')
    train, test = process_Kather_csv(root_dir)
    this_savedir_train = opj(img_savedir, 'Kather', 'train')
    this_savedir_test = opj(img_savedir, 'Kather', 'test')
    os.makedirs(this_savedir_train, exist_ok=True)
    os.makedirs(this_savedir_test, exist_ok=True)

    pbar = tqdm(total=int(len(train)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_train), X = train['image'])
    train['image'] = new_image_paths
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    train.to_csv(opj(savedir, 'Kather_train.csv'))
    test.to_csv(opj(savedir, 'Kather_test.csv'))
    
    #############################################
    #    PanNuke
    #############################################
    print('Processing PanNuke (normal, abnormal) dataset ...')
    train, test = process_PanNuke(root_dir, seed=seed, train_ratio=train_ratio)
    this_savedir_train = opj(img_savedir, 'PanNuke', 'train')
    this_savedir_test = opj(img_savedir, 'PanNuke', 'test')
    os.makedirs(this_savedir_train, exist_ok=True)
    os.makedirs(this_savedir_test, exist_ok=True)
    
    pbar = tqdm(total=int(len(train)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_train), X = train['image'])
    train['image'] = new_image_paths
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    train.to_csv(opj(savedir, 'PanNuke_train.csv'))
    test.to_csv(opj(savedir, 'PanNuke_test.csv'))
    
    #############################################
    #    DigestPath2019 (binary)
    #############################################
    
    print('Processing DigestPath (normal, abnormal) dataset ...')
    train, test = process_DigestPath(root_dir, seed=seed, train_ratio=train_ratio)
    this_savedir_train = opj(img_savedir, 'DigestPath', 'train')
    this_savedir_test = opj(img_savedir, 'DigestPath', 'test')
    os.makedirs(this_savedir_train, exist_ok=True)
    os.makedirs(this_savedir_test, exist_ok=True)
    
    pbar = tqdm(total=int(len(train)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_train), X = train['image'])
    train['image'] = new_image_paths
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    train.to_csv(opj(savedir, 'DigestPath_train.csv'))
    test.to_csv(opj(savedir, 'DigestPath_test.csv'))


    #############################################
    #    WSSS4LUAD (binary: tumor, normal)
    #############################################
    
    print('Processing WSSS4LUAD (binary: tumor, normal) dataset ...')
    train, test = process_WSSS4LUAD_binary(root_dir, seed=seed, train_ratio=train_ratio)
    this_savedir_train = opj(img_savedir, 'WSSS4LUAD', 'train')
    this_savedir_test = opj(img_savedir, 'WSSS4LUAD', 'test')
    os.makedirs(this_savedir_train, exist_ok=True)
    os.makedirs(this_savedir_test, exist_ok=True)
    
    pbar = tqdm(total=int(len(train)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_train), X = train['image'])
    train['image'] = new_image_paths
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    train.to_csv(opj(savedir, 'WSSS4LUAD_binary_train.csv'))
    test.to_csv(opj(savedir, 'WSSS4LUAD_binary_test.csv'))

    #############################################
    #    KIMIA_Path24
    #############################################
    print('Processing KIMIA_Path24 (img2img retrieval) ...')
    test = process_KIMIA_Path24(root_dir)
    this_savedir_test = opj(img_savedir, 'KIMIA_Path24', 'test')
    os.makedirs(this_savedir_test, exist_ok=True)
    
    pbar = tqdm(total=int(len(test)))
    new_image_paths = parmap(lambda fp: resizeimg(fp, this_savedir_test), X = test['image'])
    test['image'] = new_image_paths
    test.to_csv(opj(savedir, 'KIMIA_Path24_test.csv'))