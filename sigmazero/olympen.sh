#!/usr/bin/env bash

repo=~/tjack
dir=~/sigma_$(date +"%FT%T")
model=model.pt
log=log.txt

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

mkdir -p $dir
cd $dir
command $@ | xargs 2> >(tee $log >&2)
