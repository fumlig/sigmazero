#!/usr/bin/env bash

while true
do
	for f in "$@"
	do
		read -r line < $f
		echo $line
	done
done
