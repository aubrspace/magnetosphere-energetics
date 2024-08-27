#!/bin/bash
#'parallel' script for spawning lots of jobs

#define variables
INPUTDIR=./run_MEDnMEDu/GM/IO2/
#OUTPUTDIR=./outputs_ellipsoid_dist_MEDnMEDu/
#OUTPUTDIR=./outputs_r3_x120_b07_dist_MEDnMEDu/
#OUTPUTDIR=./outputs_r4_x20_b07_dist_MEDnMEDu/
#OUTPUTDIR=./outputs_r5_x150_b07_dist_MEDnMEDu/
#OUTPUTDIR=./outputs_r2625_x120_b01_dist_MEDnMEDu/
OUTPUTDIR=./outputs_r275_x120_b14_dist_MEDnMEDu/
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
    day=${file:${#INPUTDIR}+${#head}:2}
    hour=${file:${#INPUTDIR}+${#head}+3:2}
    minute=${file:${#INPUTDIR}+${#head}+5:2}
    
    # Filter by day/hour/minute
    if [[ ${day#0} -eq 7 ]] && [[ ${hour#0} -lt 8 ]]
    #if [[ 1 == 1 ]]
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

