import logging
from typing import List


class FineTuner:
    def __init__(self, alpha, seed=7):
        self.alpha = alpha
        self.seed = seed

    logging.info("FineTuner running")

    def train_and_test(self, train_x: List, train_y: List, test_x: List, test_y: List):
        pass