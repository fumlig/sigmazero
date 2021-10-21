# Sigma Zero

## Selfplay

A selfplay process should write replays to standard out, line by line. It should take one argument, a file path to the model file to use for self play. This file should be watched for changes and loaded regularly.

## Training

A training process should read replays from standard in, line by line. It should take one argument, a file path to the model file to use for training. This file should be loaded once and then written to regularly while training. When the model has been updated, the process should write a new line to standard out to indicate this.

## Running

To run locally:

```sh
MODEL=<path to save model to>
./selfplay $MODEL | ./training $MODEL
```

The training process will save its latest model as *model.pt* and receive replays from the selfplay process, which plays with the lastest model.

To run in a cluster:

```sh
CONFIG=<path to config of master and slaves>
./master.py $CONFIG
```
