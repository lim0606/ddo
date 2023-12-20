"""
Copy and modified from
https://github.com/clovaai/stargan-v2/blob/master/download.sh
"""

FILE=$1
DATADIR=$2

if [ -z $DATADIR ]; then
    echo "You must specify an output directory"
    echo "Usage example: bash download.sh afhq-v2-dataset ~/datasets"
    exit 1

fi

if  [ $FILE == "afhq-dataset" ]; then
    URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
    ZIP_FILE=${DATADIR}/afhq.zip
    mkdir -p $DATADIR
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "afhq-v2-dataset" ]; then
    #URL=https://www.dropbox.com/s/scckftx13grwmiv/afhq_v2.zip?dl=0
    URL=https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
    ZIP_FILE=${DATADIR}/afhq_v2.zip
    mkdir -p $DATADIR
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d $DATADIR
    rm $ZIP_FILE

else
    echo "Available arguments are afhq-dataset and afhq-v2-dataset."
    exit 1

fi
