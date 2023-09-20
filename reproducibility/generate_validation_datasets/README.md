# Validation Dataset Generation Instructions

This readme provides instructions on how to obtain validation datasets for your project. Follow these steps to prepare the validation datasets for the `DigestPath` and `PanNuke` datasets.

## 1. Pre-processing External Validation Datasets

### For the `DigestPath` Dataset:

To obtain the validation datasets for the `DigestPath` dataset, follow these three pre-processing steps:

1. Run the first pre-processing step:
   ```
   python preprocess/preprocess_DigestPath.py --step 1
   ```

2. Run the second pre-processing step:
   ```
   python preprocess/preprocess_DigestPath.py --step 2
   ```

3. Run the third pre-processing step:
   ```
   python preprocess/preprocess_DigestPath.py --step 3
   ```

We have separated these steps to ensure robust execution of the pre-processing code.

### For the `PanNuke` Dataset:

To obtain the validation dataset for the `PanNuke` dataset, follow this pre-processing step:
```
python preprocess/preprocess_PanNuke.py
```

## 2. Generating External Validation Datasets

For all datasets, you can generate the four external validation datasets by simply calling the following script:

```
python prepare_dataset_to_csv.py
```

These external validation datasets will be ready for use in your project after running this script.

Ensure that you have the necessary dependencies and data files available before executing the pre-processing and dataset preparation scripts.