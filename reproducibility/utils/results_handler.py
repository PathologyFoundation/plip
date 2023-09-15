import os
import pandas as pd
import logging
class ResultsHandler:

    def __init__(self, dataset, task, storing_params: dict):
        """

        :param storing_params: Dict of additional parameters we want to save in the csv
        """
        self.dataset = dataset
        self.task = task
        self.storing_params = storing_params
        logging.info("ResultsHandler created")

    def add(self, results):

        for record in results:
            record.update(self.storing_params)

        results_save_folder = os.environ["PC_RESULTS_FOLDER"]
        results_file = os.path.join(results_save_folder, f"extended_results_{self.task}_{self.dataset}.csv")
        df = pd.DataFrame(results)

        if os.path.exists(results_file):
            all_df = pd.read_csv(results_file, index_col=0)
            all_df = pd.concat([all_df, df])
            all_df.to_csv(results_file)
        else:
            df.to_csv(results_file)
        logging.info("ResultsHandler added results")

