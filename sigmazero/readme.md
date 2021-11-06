# Sigma Zero

- Selfplay will likely be the bottleneck: one GPU transfer is required per move and it seems difficult to batch.
- Processes and threads are both cheap in Linux: might as well use multiple selfplay processes on one machine instead of worrying about threads.
- Trainers and selfplayers should be both be distributable and easily testable locally.
- A single orchestrator is likely needed to distribute replays across trainers and merge their models.

## Selfplay

A selfplay process should write replays to standard out, line by line. It should take one argument, a file path to the model file to use for self play. This file should be watched for changes and loaded regularly.

## Training

A training process should read replays from standard in, line by line. It should take one argument, a file path to the model file to use for training. This file should be loaded once and then written to regularly while training. When the model has been updated, the process should write a new line to standard out to indicate this.

## Running

To run locally:

```sh
# path to store model at
MODEL=model.pt

# one selfplayer, one trainer
./selfplay $MODEL | ./training $MODEL

# three selfplayers, one trainer
{./selfplay $MODEL & ./selfplay $MODEL & ./selfplay $MODEL} | ./training $MODEL
```

The training process will save its latest model as *model.pt* and receive replays from the selfplay process, which plays with the lastest model.

To run in a cluster:

```sh
CONFIG=<path to config of master and slaves>
./master.py $CONFIG
```
