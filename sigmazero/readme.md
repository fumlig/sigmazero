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

Run one selfplayer and one trainer:

```bash
model=model.pt
./selfplay $model | ./training $model
```

Run multiple selfplayers and one trainer:

```bash
model=model.pt
./training $model <(./selfplay $model) <(./selfplay $model) <(./selfplay $model)
```

Olympen example:

```bash
./olympen.sh 01 01 02 02 03 03
```

This will create a directory `~/sigma_{datetime}` to which the latest model, checkpoints and logs will be written. The current olympen computer will be used as trainer and selfplay processes will be started on all hosts specified as arguments (NM is olympen1-1NM). The tjack repo is assumed to be at `~/tjack` but its location can be set with the `SIGMA_REPO` variable. The training session directory can be set with `SIGMA_DIR`.
