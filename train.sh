#!/bin/bash
# First get the datasets as a list
datasets=( shakespeare wikipedia philosophy linux midi game-of-thrones )

# Then loop through the list and train a model for each dataset.
for dataset in "${datasets[@]}"
do
    echo "Training model for dataset: $dataset"
    python char.py           \
      --data-source $dataset \
      --n-epochs 50	     \
      --batch-size 256       \
      --block-size 128       \
      --seed 42
done
