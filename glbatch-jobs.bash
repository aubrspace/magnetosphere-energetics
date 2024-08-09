#!/bin/bash
#'parallel' script for spawning lots of jobs

#define variables
INPUTDIR=./run_HIGHnMEDu/GM/IO2/
OUTPUTDIR=./outputs_may6_HIGHnMEDu/
#INPUTDIR=./starlink2/IO2/

# May6 Todo list:
# HighMed

filecount=0
workercount=0

#head=3d__var_1_e20140410-080
#head=3d__var_1_e201404
#head=3d__var_1_e202202
head=3d__var_1_e202206
#head=3d__var_1_e201505
#head=it2206
#head=3d__var_1_e202202

#satpath=star2satloc

i=0
#execute script on tecplot output files
for file in $INPUTDIR$head*.plt
do
    #submit a job with the following flags:
    #   -i input directory
    #   -o output directory
    #   -f specific file to process
    #echo "-i $INPUTDIR -o $OUTPUTDIR -f ${file:${#INPUTDIR}} -s $satpath"
    sbatch batchjob_energetics.gl -i $INPUTDIR -o $OUTPUTDIR \
                                  -f ${file:${#INPUTDIR}}    \
    #                              -s $satpath
    #sbatch batchjob_park.gl -f ${file:${#INPUTDIR}}

    #if you only want to process one file use this
    #i=$((i+1))
    #if [ $i == 3 ]
    #then
    #    exit
    #fi
done

