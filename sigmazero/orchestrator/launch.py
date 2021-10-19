#!/usr/bin/env python3


import os
import subprocess
import typing
import time
import yaml



with open(config_path, 'r') as f:
	config = yaml.safe_load(f)


replays_path = "./replays.fifo"


def selfplay(user: str, host: str, command: str, replays_write: typing.IO):
	args = ["ssh", f"{user}@{host}", command]
	process = subprocess.Popen(args, stdout=replays_write, stdin=subprocess.PIPE)
	return process

def training(command: str, replays_read: typing.IO):
	args = [command]
	process = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=replays_read)
	return process

r, w = os.pipe()
with open(r, "r") as replays_read, open(w, "w") as replays_write:

	selfplay_process = selfplay("oskar", "127.0.0.1", "/home/oskar/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/selfplay", replays_write)
	print("selfplay")	

	training_process = training("/home/oskar/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/training", replays_read)

	while True:
		line = training_process.stdout.readline().strip()
		print("line:", line)