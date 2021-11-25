#!/usr/bin/env bash

ids=( 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 )

for id in ${ids[@]}
do
    host="olympen1-1$id"
    users=$(ssh $host users)
    count=$(echo $users | wc -w)
    echo "$host: $count ($users)"
done