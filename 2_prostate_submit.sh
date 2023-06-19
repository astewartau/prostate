#!/bin/bash

for MODEL_NAME in CT QSM T1 QSM-T1 SWI GRE; do
    for FOLD_ID in {0..5}; do
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID 2_prostate.slurm
    done
done
