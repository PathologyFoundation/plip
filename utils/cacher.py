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
    m.update(key)

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
    m.update(key)

    save_path = os.path.join(cache_folder, m.hexdigest())

    with open(f"{save_path}", 'wb') as f:
        np.save(f, npa)
