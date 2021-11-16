#!/usr/bin/env bash

replays=replays.fifo
model=model.pt

mkfifo $replays

linelock(){
	file=$1
	while read -r line
	do
		flock -w -x $file
		echo $line >> $file
		flock -u $file
	done
}

for i in {1..4}
do
	echo "selfplay $i"
	../build/selfplay $model | linelock $replays &
done

echo "training"
../build/training $model < $replays
