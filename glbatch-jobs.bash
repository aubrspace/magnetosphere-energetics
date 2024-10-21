#!/bin/bash
#'parallel' script for spawning lots of jobs

#define variables
INPUTDIR=./run_mothersday/GM/IO2/
OUTPUTDIR=./outputs_mothersday1/

filecount=0
workercount=0

head=3d__var_1_e202405

#satpath=star2satloc

i=0
#execute script on tecplot output files
for file in $INPUTDIR$head*.plt
do
    day=${file:${#INPUTDIR}+${#head}:2}
    hour=${file:${#INPUTDIR}+${#head}+3:2}
    minute=${file:${#INPUTDIR}+${#head}+5:2}
    
    # Filter by day/hour/minute
    #if [[ ${day#0} -eq 7 ]] && [[ ${hour#0} -lt 8 ]]
    if [[ 1 == 1 ]]
    then
        #echo "${file:${#INPUTDIR}} $day $hour"

        #submit a job with the following flags:
        #   -i input directory
        #   -o output directory
        #   -f specific file to process
        sbatch batchjob_energetics.gl -i $INPUTDIR -o $OUTPUTDIR \
                                      -f ${file:${#INPUTDIR}}    \
                                      -s $satpath
        #sbatch batchjob_park.gl -f ${file:${#INPUTDIR}}

        # If you only want to process n files from the list
        #i=$((i+1))
        #if [ $i == 3 ]
        #then
        #    exit
        #fi
        #exit

    fi
done

