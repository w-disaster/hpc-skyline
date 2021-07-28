#!/bin/bash

FILES="/home/local/STUDENTI/luca.fabri/datafiles/test*"
for filename in $FILES; do
	stdout=$(echo $filename | cut -d'/' -f 7 | cut -d'.' -f 1)
	echo $stdout
	./v4 < $filename > ./cuda_speedup/${stdout}.txt
done


