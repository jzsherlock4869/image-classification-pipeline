PYTHONNAME=/path/to/your/python

EXPNAME=$(cat basic_config.py | grep EXP_NAME | awk -F"'" '{print $2}')
OUTDIR=$(cat basic_config.py | grep OUTPUT_DIR | awk -F"'" '{print $2}')
echo "run experiment : "$EXPNAME
echo $OUTDIR
if [ ! -d $OUTDIR ]
then
    mkdir $OUTDIR
fi
if [ ! -d $OUTDIR/$EXPNAME ]
then
    mkdir $OUTDIR/$EXPNAME
    cp basic_config.py $OUTDIR/$EXPNAME/basic_config.py
    $PYTHONNAME -u train_classification.py 2>&1 | tee tr.log $OUTDIR/$EXPNAME/train.log > /dev/null &
else
    echo "experiment name aliased, pleas change EXP_NAME in basic_config.py !"
fi

