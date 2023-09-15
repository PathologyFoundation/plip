import os
import hashlib
import numpy as np


def get_cache_name(name: str, path: str):
    """
    Generates the cache name of the file using sha256.
    :param name:
    :param path:
    :return:
    """

    key = name+path

    cache_folder = os.environ["PC_CACHE_FOLDER"]
    m = hashlib.sha256()
    m.update(key.encode('utf-8'))

    save_path = os.path.join(cache_folder, m.hexdigest())

    return save_path

def cache_hit_or_miss(name: str, path: str):
    save_path = get_cache_name(name, path)

    if os.path.exists(save_path):
        return np.load(save_path)
    else:
        return None


def cache_numpy_object(npa, name, path):
    key = name+path

    cache_folder = os.environ["PC_CACHE_FOLDER"]
    m = hashlib.sha256()
    m.update(key.encode('utf-8'))

    save_path = os.path.join(cache_folder, m.hexdigest())

    with open(f"{save_path}", 'wb') as f:
        np.save(f, npa)



###############################################################
# below are new codes
###############################################################
def get_savepath(name, path):
    modelname, dataset_name = name.split('img')
    dataset_name = dataset_name.split('.csv')[0]
    cache_folder = os.environ["PC_CACHE_FOLDER"]
    cache_subfolder_data = os.path.join(cache_folder, dataset_name, modelname)
    os.makedirs(cache_subfolder_data, exist_ok=True)
    if modelname == 'plip':
        path = os.path.basename(path)
    save_path = os.path.join(cache_subfolder_data, path)
    return save_path

def cache_hit_or_miss_raw_filename(name: str, path: str):
    save_path = get_savepath(name, path)
    if os.path.exists(save_path):
        print('[CACHE] Found existed embedding.')
        return np.load(save_path)
    else:
        print('[CACHE] No existed embedding found. Need to generate embedding first.')
        return None

def cache_numpy_object_raw_filename(npa, name, path):
    save_path = get_savepath(name, path)
    print(f"[CACHE] Saving embedding. Name: {name}, Path: {path}, Save path: {save_path}")
    with open(f"{save_path}", 'wb') as f:
        np.save(f, npa)
