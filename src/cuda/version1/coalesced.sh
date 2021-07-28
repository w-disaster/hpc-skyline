#!/bin/bash

FILES="/home/local/STUDENTI/luca.fabri/datafiles/test9*"
for filename in $FILES; do
	stdout=$(echo $filename | cut -d'/' -f 7 | cut -d'.' -f 1)
	echo $stdout
	./v1 < $filename > coalesced/v1_${stdout}.txt
	./v2 < $filename > coalesced/v2_${stdout}.txt
	./v3 < $filename > coalesced/v3_${stdout}.txt
	./v4 < $filename > coalesced/v4_${stdout}.txt
done

