#!/bin/bash
#Batch file for running extract_mpsurface.py on all .plt files

#set path to .idl files and script
IDLDIR=./ionosphere_data/15minute/
SCRIPTDIR=./

#execute script on .idl files
for file in $IDLDIR*.idl
do
    python $SCRIPTDIR/idl2pandas.py $file $IDLDIR 
done
