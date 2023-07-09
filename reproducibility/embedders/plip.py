import clip
import tqdm
import numpy as np
import torch
from reproducibility.embedders.internal_datasets import *
from torch.utils.data import DataLoader
from reproducibility.utils.cacher import cache_hit_or_miss, cache_numpy_object, cache_hit_or_miss_raw_filename, cache_numpy_object_raw_filename

class CLIPEmbedder:

    def __init__(self, model, preprocess, name, backbone):
        self.model = model
        self.preprocess = preprocess
        self.name = name
        self.backbone = backbone
        
    def image_embedder(self, list_of_images, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss_raw_filename(self.name + "img" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_images(list_of_images, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object_raw_filename(hit, self.name + "img" + additional_cache_name, self.backbone)
            return hit

    def text_embedder(self, list_of_labels, device="cuda", num_workers=1, batch_size=32, additional_cache_name=""):
        hit_or_miss = cache_hit_or_miss(self.name + "txt" + additional_cache_name, self.backbone)

        if hit_or_miss is not None:
            return hit_or_miss
        else:
            hit = self.embed_text(list_of_labels, device=device, num_workers=num_workers, batch_size=batch_size)
            cache_numpy_object(hit, self.name + "txt" + additional_cache_name, self.backbone)
            return hit

    def embed_images(self, list_of_images, device="cuda", num_workers=1, batch_size=32):
        train_dataset = CLIPImageDataset(list_of_images, self.preprocess)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

        image_embeddings = []

        total = len(list_of_images) // batch_size
        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for images in dataloader:
                images = images.to(device)
                image_embeddings.extend(self.model.encode_image(images).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()

        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        return image_embeddings

    def embed_text(self, list_of_labels, device="cuda", num_workers=1, batch_size=32):
        train_dataset = CLIPCaptioningDataset(list_of_labels)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        text_embeddings = []
        total = len(list_of_labels) // batch_size

        pbar = tqdm.tqdm(total=total, position=0)
        with torch.no_grad():
            for captions in dataloader:
                idx = clip.tokenize(captions, truncate=True).to(device)
                text_embeddings.extend(self.model.encode_text(idx).detach().cpu().numpy())

                pbar.update(1)

            pbar.close()

        text_embeddings = np.array(text_embeddings)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        return text_embeddings


