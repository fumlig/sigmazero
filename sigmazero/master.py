#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import yaml
import colorama
import torch
import heapq


# todo:

# each slave should be able to have multiple selfplay and training processes
# could it be possible to keep models in memory?

# would be nice to have models in memory...

# the C++ API (and inferences) requires knowledge of the architecture
# for the forward method, we might be able to solve it by having the model
# take a position as input

# selfplayers and trainers can actually be load balanced pretty easily
# 


MODEL_NAME = "model.pt"

SELFPLAY_COST = (1, 0) # 1 cpu
TRAINING_COST = (1, 1) # 1 cpu, 1 gpu

SELFPLAY_COLOR = colorama.Fore.BLUE
TRAINING_COLOR = colorama.Fore.CYAN

INFO_COLOR = colorama.Fore.GREEN
WARNING_COLOR = colorama.Fore.YELLOW
ERROR_COLOR = colorama.Fore.RED


def prefix_pipe(pipe_write: int, prefix: str):
	# return a file descriptor for which written lines will be prefixed with a string
	prefix_read, prefix_write = os.pipe()

	subprocess.Popen(
		["sed", "-e", f"s/^/{prefix}/;"],
		stdin=prefix_read,
		stdout=pipe_write,
		stderr=None,
	)

	return prefix_write


def print_info(message: str):
	print(INFO_COLOR + "master|info: " + colorama.Fore.RESET + message)

def print_warning(message: str):
	print(WARNING_COLOR + "master|warning: " + colorama.Fore.RESET + message)

def print_error(message: str):
	print(ERROR_COLOR + "master|error: " + colorama.Fore.RESET + message)


class Slave(object):

	def __init__(
		self,
		selfplay: str,
		training: str,
		user: str = None,
		host: str = "127.0.0.1",
		name: str = None,
		cpus: int = 1,
		gpus: int = 0,
		directory: str = "/tmp",
		authorize: bool = False
	):
		self.user = user
		self.host = host	
		self.name = name if name else self.destination

		self.resources = self.available = (cpus, gpus)
		
		self.selfplay = selfplay
		self.training = training
		self.directory = directory

		if authorize:
			self.authorize()

		self.trainers = []
		self.selfplayers = []


	def __cmp__(self, other):
		assert isinstance(other, Slave)
		return cmp(self.available, other.available, )


	@property
	def destination(self):
		if self.user:
			return f"{self.user}@{self.host}"
		else:
			return self.host

	def command(self, command: str, arguments: list = None, stdout: int = subprocess.PIPE, stdin: int = subprocess.PIPE, stderr: int = None, wait: bool = False) -> subprocess.Popen:
		process = subprocess.Popen(
			["ssh", self.destination, command] + (arguments if arguments else []),
			stdout=stdout, stdin=stdin, stderr=stderr
		)

		if wait:
			process.wait()

		return process

	def upload(self, source: str, target: str, recursive: bool = False):
		subprocess.run(
			["scp"] + (["-r"] if recursive else []) + [source, f"{self.destination}:{target}"],
			check=True,
			capture_output=True,
		)

	def download(self, source: str, target: str, recursive: bool = False):
		subprocess.run(
			["scp"] + (["-r"] if recursive else []) + [f"{self.destination}:{source}", target],
			check=True,
			capture_output=True,
		)

	def authorize(self):
		subprocess.run(
			["ssh-copy-id", self.destination],
			check=True,
			capture_output=True
		)

	def spawn_training(self, replays_read: int) -> bool:
		command = self.training
		arguments = [os.path.join(self.directory, MODEL_NAME)]		
		identifier = self.used[0] + self.used[1]

		if not self.consume(TRAINING_COST):
			return False

		prefix = TRAINING_COLOR + f"{self.name}|training{identifier}: " + colorama.Fore.RESET
		prefix_write = prefix_pipe(None, prefix)

		self.trainers.append(self.command(command, arguments, stdin=replays_read, stderr=prefix_write))

	def spawn_selfplay(self, replays_write: int) -> bool:
		command = self.selfplay
		arguments = [os.path.join(self.directory, MODEL_NAME)]
		identifier = self.used[0] + self.used[1]
		
		if not self.consume(SELFPLAY_COST):
			return False

		prefix = SELFPLAY_COLOR + f"{self.name}|selfplay{identifier}: " + colorama.Fore.RESET		
		prefix_write = prefix_pipe(None, prefix)

		self.selfplayers.append(self.command(command, arguments, stdout=replays_write, stderr=prefix_write))
		
		return True


	def upload_model(self, source_path: str):
		self.upload(source_path, self.model_path)

	def download_model(self, target_path: str):
		self.download(self.model_path, target_path)

	def consume(cost: (int, int)) -> False:
		if self.available[0] - cost[0] <= 0 or self.available[1] - cost[1]:
			return False

		self.available[0] -= cost[0]
		self.available[1] -= cost[1]

		return True

	def respawn(self, respawn=False):
		for trainer in self.trainers:
			if trainer.poll() is None:
				print_warning(f"{self.name} has dead trainer")
				self.spawn_selfplay(trainer.stdout)
				self.trainers.remove(trainer)
				self.used[0] -= TRAINING_COST[0]
				self.used[1] -= TRAINING_COST[1]
				print_info(f"{self.name} trainer respawned")

		for selfplayer in self.selfplayers:
			if selfplayer.poll() is None:
				print_warning(f"{self.name} has dead selfplayer")
				self.spawn_selfplay(selfplayer.stdout)
				self.selfplayers.remove(selfplayer)
				self.used[0] -= SELFPLAY_COST[0]
				self.used[1] -= SELFPLAY_COST[1]
				print_info(f"{self.name} selfplayer respawned")


if __name__ == "__main__":

	colorama.init()

	parser = argparse.ArgumentParser(description="Sigma Zero master script")
	parser.add_argument("config", type=str, help="Path to configuration file")

	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	directory = config["directory"]
	checkpoints = config["checkpoints"]
	trainers = config["trainers"]
	selfplayers = config["selfplayers"]

	replays_read, replays_write = os.pipe()

	trainers_left = trainers
	selfplayers_left = selfplayers

	heapq.heapify(slaves)

	for slave in :
		while slave.spawn_training(replays_read) or slave.spawn_selfplay(replays_write):
			pass


	while True:
		models = []

		for trainer, process in trainers:
			if process.stdout.readline():
				trainer.download_model(model_path)
				print_info(f"downloaded model from trainer {trainer.destination}")

				model = torch.jit.load(model_path)
				models.append(model)

		if not models:
			continue

		# todo:
		# eventually, merging of models can be done here
		# also checkpoints, evaluation etc.
		model = models[0]
		
		print(model.__dict__)

		torch.jit.save(model, model_path)
		print_info(f"merged {len(models)} model(s)")

		for selfplayer, _ in selfplayers:
			selfplayer.upload_model(model_path)
			print_info(f"uploaded model to selfplayer {selfplayer.destination}")

	colorama.deinit()
