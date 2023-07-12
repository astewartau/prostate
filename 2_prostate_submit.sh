#!/bin/bash

#for MODEL_NAME in CT QSM T1 QSM-T1 QSM-T1-T2s SWI GRE; do
#for MODEL_NAME in SWI GRE; do
for MODEL_NAME in R2s QSM-T1-R2s; do
    for FOLD_ID in {0..23}; do
        echo $MODEL_NAME-$FOLD_ID
        sbatch --job-name="${MODEL_NAME}-${FOLD_ID}" --export=ALL,MODEL_NAME=$MODEL_NAME,FOLD_ID=$FOLD_ID 2_prostate.slurm
    done
done
