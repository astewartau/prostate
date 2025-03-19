#!/bin/bash

#for MODEL_NAME in FMAP QSM-FMAP QSM SWI T1 R2s GRE QSM-T1-R2s QSM-T1 QSM-SWI; do
for MODEL_NAME in T1; do
    for FOLD_ID in {6,20,25}; do
        echo $MODEL_NAME-$FOLD_ID
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID train_one_model.slurm
    done
done

