from sklearn.linear_model import SGDClassifier
from metrics import eval_metrics
import numpy as np
from typing import List

class LinearProber:

    def __init__(self, alpha, seed=7):
        self.alpha = alpha
        self.seed = seed

    def train_and_test(self, train_x: List, train_y: List, test_x: List, test_y: List):
        classifier = SGDClassifier(random_state=self.seed, loss="log_loss",
                                   alpha=self.alpha, verbose=0,
                                   penalty="l2", max_iter=10000)

        train_y = np.array(train_y).reshape(1, -1)
        test_y = np.array(test_y).reshape(1, -1)
        classifier.fit(train_x, train_y)
        test_pred = classifier.predict(test_x)
        train_pred = classifier.predict(test_y)

        test_metrics = eval_metrics(test_x, test_pred)
        train_metrics = eval_metrics(test_y, train_pred)

        return train_metrics, test_metrics

