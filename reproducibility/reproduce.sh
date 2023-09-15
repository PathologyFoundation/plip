cd scripts
datasets=("Kather" "PanNuke" "DigestPath" "WSSS4LUAD_binary")

# Reproducing linear probing experiments
# Iterate over different datasets and regularization strengths
for dataset in "${datasets[@]}";
do
  for alpha in 0.0001 0.001 0.01 0.1;
  do
    python3 linear_probing_evaluation.py --model_name="plip" --alpha=$alpha --dataset $dataset
    python3 linear_probing_evaluation.py --model_name="mudipath" --alpha=$alpha --dataset $dataset
    python3 linear_probing_evaluation.py --model_name="clip" --alpha=$alpha --dataset $dataset
  done
done

# Reproducing zero-shot experiments

# TODO

# Reproducing text-to-image retrieval experiments

# TODO

