#!/bin/bash

#FILES="/home/local/STUDENTI/luca.fabri/datafiles/test*"
FILES="test5-N100000-D20.in test6-N100000-D50.in test7-N100000-D200.in"
for filename in $FILES; do
	filename="/home/local/STUDENTI/luca.fabri/datafiles/${filename}"
	stdout=$(echo $filename | cut -d'/' -f 7 | cut -d'.' -f 1)
	echo $stdout
	./v1-benchmark < $filename > "v1_stdout/v1-${stdout}.txt"
done
