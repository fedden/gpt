#!/bin/bash
# First get the datasets as a list
datasets=( shakespeare wikipedia philosophy linux midi game-of-thrones )
contexts=( 
    "I dispair! Thou art the cause, for thou hast stol'n my heart."
    "Leon Fedden is a " 
    "What is real anyway? "
    "/***************"
    "01 00 FF 2F 00"
    "The king exclaimed, "
)

# Then loop through the dataset and context list and train a model for each dataset.
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    context=${contexts[$i]}
    echo "Training model for dataset: $dataset"
    python char.py           \
      --data-source $dataset \
      --n-epochs 50	     \
      --batch-size 256       \
      --block-size 128       \
      --seed 42
done
