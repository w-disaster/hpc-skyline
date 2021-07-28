#!/bin/bash

#FILES="/home/local/STUDENTI/luca.fabri/datafiles/weak_scaling/*"
#for filename in $FILES; do
#	stdout=$(echo $filename | cut -d'/' -f 7 | cut -d'.' -f 1)
#	echo $stdout
#	OMP_NUM_THREADS=./v3 < $filename > weakscaling/${stdout}.txt
#done

for I in {1..12}; do
	FILES="/home/local/STUDENTI/luca.fabri/weak_scaling/*t${I}-*"
	for FILE in $FILES; do
		STDOUT=$(echo $FILE | cut -d'/' -f 7 | cut -d'.' -f 1)
		echo $STDOUT
		OMP_NUM_THREADS=${I} ./omp-skyline < $FILE > weakscaling/${STDOUT}.txt
	done
done
