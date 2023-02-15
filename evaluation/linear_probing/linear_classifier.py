from sklearn.linear_model import SGDClassifier
from metrics import eval_metrics
import numpy as np
from typing import List
from sklearn.preprocessing import LabelEncoder
import logging

class LinearProber:

    def __init__(self, alpha, seed=7):
        self.alpha = alpha
        self.seed = seed

    logging.info("LinearProber running")

    def train_and_test(self, train_x: List, train_y: List, test_x: List, test_y: List):
        classifier = SGDClassifier(random_state=self.seed, loss="log_loss",
                                   alpha=self.alpha, verbose=0,
                                   penalty="l2", max_iter=10000, class_weight="balanced")

        le = LabelEncoder()

        train_y = le.fit_transform(train_y)
        test_y = le.transform(test_y)

        train_y = np.array(train_y)
        test_y = np.array(test_y)

        classifier.fit(train_x, train_y)
        test_pred = classifier.predict(test_x)
        train_pred = classifier.predict(train_x)

        test_metrics = eval_metrics(test_y, test_pred)
        train_metrics = eval_metrics(train_y, train_pred)
        print(train_metrics)
        print()
        print(test_metrics)
        logging.info(f"LinearProber Results on Test")
        logging.info(str(test_metrics))

        return train_metrics, test_metrics

