from reproducibility.metrics import retrieval_metrics
import logging

class ImageRetrieval:

    def __init__(self):
        pass

    def retrieval(self, image_embeddings, text_embeddings):

        best_scores = []

        for t in text_embeddings:
            arr = t.dot(image_embeddings.T)

            best = arr.argsort()[-50:][::-1]

            best_scores.append(best)

        targets = list(range(0, len(image_embeddings)))

        test_metrics = retrieval_metrics(targets, best_scores)
        train_metrics = retrieval_metrics(targets, best_scores)

        test_metrics["split"] = "test"
        train_metrics["split"] = "train"

        logging.info(f"Retrieval Done")

        return train_metrics, test_metrics
