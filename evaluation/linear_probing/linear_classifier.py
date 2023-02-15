from sklearn.linear_model import SGDClassifier
from metrics import eval_metrics


class LinearProber:

    def __init__(self, alpha, seed=7):
        self.alpha = alpha
        self.seed = seed

    def train_and_test(self, train_x, train_y, test_x, test_y):
        classifier = SGDClassifier(random_state=self.seed, loss="log_loss",
                                   alpha=self.alpha, verbose=0,
                                   penalty="l2", max_iter=10000)

        classifier.fit(train_x, train_y)
        test_pred = classifier.predict(test_x)
        train_pred = classifier.predict(test_y)

        test_metrics = eval_metrics(test_x, test_pred)
        train_metrics = eval_metrics(test_y, train_pred)

        return train_metrics, test_metrics

