#!/bin/bash

FILENAME="test7-v4-"

for I in {1..32}
do
	diff test7-v1.txt test7-v4-$I
done 
