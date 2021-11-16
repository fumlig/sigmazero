#!/usr/bin/env bash

while true
do
	for f in "$@"
	do
		while read -r -t 0 line < $f
		do
			echo $line

		done
	done
done
