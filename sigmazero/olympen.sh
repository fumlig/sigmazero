#!/usr/bin/env bash

[ $# -eq 0 ] && echo "error: no slaves specified" && exit 1

repo=${SIGMA_REPO:-~/tjack}
dir=${SIGMA_DIR:-~/sigma_$(date +"%FT%T")}
model=model.pt
log=log.txt

function prefix {
	gawk -v pre="$1" '{ print "[" strftime("%Y-%m-%d %H:%M:%S") " " pre "]" , $0 ; fflush() }'
}

function selfplay {
	ssh olympen1-1$1.ad.liu.se "$repo/build/selfplay $dir/$model" 2> >(prefix "selfplay$1" >&2)
}

function training {
	$repo/build/training $model $@ 2> >(prefix "training" >&2)
}

function command
{
	eval "training $(printf '<(selfplay %s) ' $@)" 2> >(tee -a $log)
}

echo "olympen session"
echo "==============="
echo -e "repo:\t$repo"
echo -e "dir:\t$dir"
echo -e "model:\t$dir/$model"
echo -e "log:\t$dir/$log"
echo -e "slaves:\t$@"

mkdir -p $dir
cd $dir
command $@