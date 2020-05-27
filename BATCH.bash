#!/bin/bash
#Batch file for running extract_mpsurface.py on all .plt files

#create output folders
PLTDIR=plt/
SCRIPTDIR=./
mkdir output
mkdir output/plt
mkdir output/lay
mkdir output/png
PLTOUT=output/plt/
LAYOUT=output/lay/
PNGOUT=output/png/
echo "Created output directory"

#ensure batch mode is ready for tecpot LINUX SPECIFIC
TECPATH=$(which tec360)
eval `$TECPATH-env`
echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH

#create log file for integral quantities
touch integral_log.csv
echo $'year, month, day, hour, minute, second, abstime, k_flux[kW],'>>integral_log.csv

#execute script on .plt files
for file in $PLTDIR*.plt
do
    python extract_mpsurface.py $file $PLTOUT $LAYOUT $PNGOUT
done
