import numpy as np
from embedders.abst import AbstractEmbedder
from metrics import retrieval_metrics
import logging

class ImateRetrieval:

    def __init__(self):
        pass

    def retrieval(self, image_embeddings, text_embeddings):

        best_scores = []

        for t in text_embeddings:
            arr = t.dot(image_embeddings.T)

            best = arr.argsort()[-10:][::-1]

            best_scores.append(best)

        test_metrics = retrieval_metrics(list(range(0, len(image_embeddings))))

        test_metrics["split"] = "test"

        logging.info(f"Retrieval Results on Test")
        logging.info(str(test_metrics))

        return test_metrics
