import numpy as np
from embedders.abst import AbstractEmbedder


class ZeroShotClassifier:

    def __init__(self):
        pass

    def zero_shot_classification(self, image_embeddings, text_embeddings, labels):

        score = image_embeddings.dot(text_embeddings.T)
        predictions = [labels[np.argmax(i)] for i in score]

        return predictions
