# Sigma Zero

## Running

To run locally:

```sh
MODEL=<path to save model to>
./selfplay $MODEL | ./training $MODEL
```

The training process will save its latest model as *model.pt* and receive replays from the selfplay process, which plays with the lastest model.

To run in a cluster:

```
CONFIG=<path to config of master and slaves>
./master.py --config $CONFIG
```
