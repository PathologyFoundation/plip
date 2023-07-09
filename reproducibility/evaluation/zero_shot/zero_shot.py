import numpy as np
from reproducibility.metrics import eval_metrics
import logging

class ZeroShotClassifier:

    def __init__(self):
        pass

    def zero_shot_classification(self, image_embeddings, text_embeddings, unique_labels, target_labels):

        score = image_embeddings.dot(text_embeddings.T)
        predictions = [unique_labels[np.argmax(i)] for i in score]


        test_metrics = eval_metrics(target_labels, predictions)
        train_metrics = eval_metrics(target_labels, predictions)
        test_metrics["split"] = "test"
        train_metrics["split"] = "train"

        import pickle
        with open("pickle.pkl", "wb") as f:
            pickle.dump({"target" : target_labels,
                         "predictions" : predictions}, f)
        exit()
        logging.info(f"ZeroShot Done")

        return train_metrics, test_metrics
