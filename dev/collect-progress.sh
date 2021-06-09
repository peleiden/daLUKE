#!/usr/bin/sh
MODELPATH=/work3/s183912/pdata2/johnny-charlie
OUTPATH=/work3/s183911/progress

FILES=`ls $MODELPATH/daluke_epoch*.pt`
for F in $FILES
do
    EPOCH=`echo $F | cut -c 50-$((${#F}-3))`
    echo $EPOCH
    python -m daluke.collect_modelfile $F "${OUTPATH}/main_epoch${EPOCH}.tar.gz"
done
