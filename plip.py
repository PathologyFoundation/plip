import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import PIL
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset, Image


class PLIP:


    def __init__(self, model_name, auth_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess, self.model_hash = self._load_model(model_name, auth_token=auth_token)
        self.model = self.model.to(self.device)


    def _load_model(self,
                    name: str,
                    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    auth_token=None):

        model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
        preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)

        return model, preprocessing, hash

    def encode_images(self, images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
        def transform_fn(el):
            imgs = el['image'] if isinstance(el['image'][0], PIL.Image.Image) else [Image().decode_example(_) for _ in
                                                                                    el['image']]
            return self.preprocess(images=imgs, return_tensors='pt')

        dataset = Dataset.from_dict({'image': images})
        dataset = dataset.cast_column('image', Image(decode=False)) if isinstance(images[0], str) else dataset
        # dataset = dataset.map(map_fn,
        #             batched=True,
        #             remove_columns=['image'])
        dataset.set_format('torch')
        dataset.set_transform(transform_fn)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        image_embeddings = []
        pbar = tqdm(total=len(images) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                image_embeddings.extend(self.model.get_image_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(image_embeddings)

    def encode_text(self, text: List[str], batch_size: int):
        dataset = Dataset.from_dict({'text': text})
        dataset = dataset.map(lambda el: self.preprocess(text=el['text'], return_tensors="pt",
                                                         max_length=77, padding="max_length", truncation=True),
                              batched=True,
                              remove_columns=['text'])
        dataset.set_format('torch')
        dataloader = DataLoader(dataset, batch_size=batch_size)
        text_embeddings = []
        pbar = tqdm(total=len(text) // batch_size, position=0)
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                text_embeddings.extend(self.model.get_text_features(**batch).detach().cpu().numpy())
                pbar.update(1)
            pbar.close()
        return np.stack(text_embeddings)

    def _cosine_similarity(self, key_vectors: np.ndarray, space_vectors: np.ndarray, normalize=True):
        if normalize:
            key_vectors = key_vectors / np.linalg.norm(key_vectors, ord=2, axis=-1, keepdims=True)
        return np.matmul(key_vectors, space_vectors.T)

    def _nearest_neighbours(self, k, key_vectors, space_vectors, normalize=True, debug=False):
        if type(key_vectors) == List:
            key_vectors = np.array(key_vectors)
        if type(space_vectors) == List:
            space_vectors = np.array(space_vectors)

        cosine_sim = self._cosine_similarity(key_vectors, space_vectors, normalize=normalize)
        nn = cosine_sim.argsort()[:, -k:][:, ::-1]

        return nn

    def zero_shot_classification(self, images, text_labels: List[str], debug=False):
        """
        Perform zero-shot image classification
        :return:
        """
        # encode text
        text_vectors = self.encode_text(text_labels, batch_size=8)
        # encode images
        image_vectors = self.encode_images(images, batch_size=8)
        # compute cosine similarity
        cosine_sim = self._cosine_similarity(image_vectors, text_vectors)
        if debug:
            print(cosine_sim)
        preds = np.argmax(cosine_sim, axis=-1)
        return [text_labels[idx] for idx in preds]

    def retrieval(self, queries: List[str], top_k: int = 10):
        """
        Image retrieval from queries
        :return:
        """
        # encode text
        text_vectors = self.encode_text(queries, batch_size=8)
        # compute cosine similarity
        # cosine_sim = self._cosine_similarity(text_vectors, self.image_vectors)
        return self._nearest_neighbours(k=top_k, key_vectors=text_vectors, space_vectors=self.image_vectors)

        # return np.argmax(cosine_sim, axis=-1)
        # return cosine_sim.argsort()[:,-top_k:][:,::-1]

