# Sigma Zero

## Training

1. Model is initialized.
2. Self play workers started.
3. Repeat until training done:
	a. Push model to workers.
	b. Read replays from workers.
	c. Train on replays.

The latest model is saved to disk. Changes to this model are watched with
inotify. When it is changed, its weights are copied to a predetermined location
on each worker. The workers are started and their replays are written to a file.
This file is continuosly read and used to update the model.

## Selfplay

1. Model is initialized.
2. Repeat until selfplay done:
	a. Load latest weights.
	b. Play a game.
	c. Write game, values and policies.

## Plan

Eventually, we can have one master that handles all distribution.
Slaves are either trainers or selfplayers. When a trainer has a new
model it is downloaded by the master, checked if it is better or worse, then uploaded to selfplayers.