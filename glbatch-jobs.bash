#!/bin/bash
#'parallel' script for spawning lots of jobs

#define variables
#INPUTDIR=./ideal_test/GM/IO2/
#OUTPUTDIR=./ideal_first_results/
INPUTDIR=./starlink2/IO2/
OUTPUTDIR=./star2_outputs_xslice/
filecount=0
workercount=0

#head=3d__var_1_e201404
head=3d__var_1_e202202

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

    #if you only want to process one file use this
    #i=$((i+1))
    #if [ $i == 4 ]
    #then
    #    exit
    #fi
done

