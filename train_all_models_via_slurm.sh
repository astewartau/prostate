#!/bin/bash

for MODEL_NAME in CT QSM SWI T1 R2s GRE QSM-T1-R2s QSM-T1 QSM-SWI; do
    for FOLD_ID in {0..25}; do
        echo $MODEL_NAME-$FOLD_ID
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID 2_prostate.slurm
    done
done

