#!/bin/sh
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -gpu "num=1"
#BSUB -J "LUKE-fine"
#BSUB -R "rusage[mem=20GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -u s183911@student.dtu.dk
#BSUB -N
#BSUB -oo ~/joblogs/stdout_%J
#BSUB -eo ~/joblogs/stderr_%J

MODELPATH=/work3/$USER/progress
OUTPATH=/work3/$USER/progress-finetune

FILES=`ls $MODELPATH/main_epoch*.tar.gz`
for F in $FILES
do
    EPOCH=`echo $F | cut -c 35-$((${#F}-7))`

    daluke/ner/run.py $OUTPATH\
        -c configs/finetune-exp.ini\
        -m $F\
        --name "epoch-$EPOCH"

    RESPATH="$OUTPATH/epoch-$EPOCH"

    python daluke/plot/plot_finetune_ner.py $RESPATH/train-results
    python daluke/ner/run_eval.py $RESPATH
done
