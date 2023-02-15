import numpy as np
from embedders.abst import AbstractEmbedder


class ZeroShotClassifier:

    def __init__(self, embedder: AbstractEmbedder, batch_size, num_workers, prompt_modifier=None):
        self.prompt_modifier = prompt_modifier
        self.embedder = embedder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def zero_shot_classification(self, images, labels, device):
        if self.prompt_modifier:
            labels = [self.prompt_modifier(k) for k in labels]

        image_embeddings = self.embedder.image_embedder(images, device, self.num_workers, self.batch_size)
        text_embeddings = self.embedder.text_embedder(labels, device, self.num_workers, self.batch_size)

        score = image_embeddings.dot(text_embeddings.T)
        predictions = [labels[np.argmax(i)] for i in score]

        return predictions
