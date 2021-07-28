#!/bin/bash

FILES="/home/local/STUDENTI/luca.fabri/datafiles/test*"
#mkdir v2-times
for filename in $FILES; do
	#filename="/home/local/STUDENTI/luca.fabri/datafiles/${filename}"
	stdout=$(echo $filename | cut -d'/' -f 7 | cut -d'.' -f 1)
	echo $stdout
	OMP_NUM_THREADS=12 ./omp-skyline-v2 < $filename > "./v2-times/${stdout}_12.txt"
done
