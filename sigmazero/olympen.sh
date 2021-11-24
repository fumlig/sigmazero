#!/usr/bin/env bash

repo=~/tjack
directory=~/sigma
model=sigmazero.pt

function prefix {
	gawk -v arg="$1" '{ print strftime("[%Y-%m-%d %H:%M:%S]" arg) , $0 }'
}

function selfplay {
	ssh olympen1-$1.ad.liu.se $repo/build/selfplay $model 2> >(prefix selfplay$1 >&2)
}

function training {
	$repo/build/training $model 2> >(prefix training >&2)
}

function command {
    echo "training"
    for i in $@
    do
        echo "<(selfplay $i)"
    done
}

mkdir -p $directory
cd $directory
command $@ | xargs
