#!/usr/bin/env bash

# https://unix.stackexchange.com/questions/102831/how-to-combine-lines-printed-by-multiple-programs-safely

computers=( 101 102 103 )
selfplay=~/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/selfplay
training=~/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/training
model=/tmp/model.pt


function hostname {
	name=$1
	echo "olympen1-$computer.ad.liu.se"
}

function training {
	name=$1
	command=$2
	prefix="training-$name: "
	mkfifo /tmp/sigmazero-training-$name
	$command $model
}

function selfplay {
	name=$1
	command=$2
	prefix="selfplay $1: "
	mkfifo /tmp/$1
}

#sed -e "s/^/$prefix/;"

#mkfifo $replays

master=>($training $model)
echo "master status: $?"

slaves=()
slaves+=(<($selfplay $model))
echo "slave status: $?"

echo "master: $master"
echo "slaves: ${slaves[@]}"

while true
do
	for slave in ${slaves[@]}
	do
		if read -r -t 0 replay < $slave
		then
			echo "replay: $replay"
		fi
	done
	sleep 1
done
