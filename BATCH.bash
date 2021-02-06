#!/bin/bash
#Batch file for running extract_mpsurface.py on all .plt files

#create output folders
PLTDIR=pltdbug/15minute/
SCRIPTDIR=./
mkdir output
mkdir output/plt
mkdir output/lay
mkdir output/png
mkdir output/figures
OUT=output/
PLTOUT=output/plt/
LAYOUT=output/lay/
PNGOUT=output/png/
echo "Created output directory"

#ensure batch mode is ready for tecpot LINUX SPECIFIC
TECPATH=$(which tec360)
eval `$TECPATH-env`
echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH

#execute script on .plt files
for file in $PLTDIR*.plt
do
    python main.py $file $OUT $PNGOUT $PLTOUT >> $OUT/outlog.log
done

#create video from png file
python global_energetics/makevideo.py $PNGOUT
mv $PNGOUT/frames/video.avi $OUT

#create figures of integrated quantities over time
python global_energetics/mpdynamics_analysis/proc_temporal.py
