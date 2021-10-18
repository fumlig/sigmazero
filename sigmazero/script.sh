config=$1

[[ ! -f $config ]] && echo "invalid/no config specified" && exit 1

replays=$(yq -r ".master.replays" $config)

[[ -e $replays ]] && rm $replays
mkfifo $replays

# for now: this should be sent to trainers
cat $replays &

n=$(yq ".slaves | length" $config)

for i in $(seq 0 $(($n-1)))
do
	name=$(yq -r ".slaves[${i}].name" $config)
	user=$(yq -r ".slaves[${i}].user" $config)
	host=$(yq -r ".slaves[${i}].host" $config)
	type=$(yq -r ".slaves[${i}].type" $config)
	executable=$(yq -r ".slaves[${i}].executable" $config)
	
	# todo: passwords and race conditions
	ssh ${user}@${host} "${executable}" > $replays &
	pid=$!

	echo "$name running with pid $pid"
done