#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import yaml
import colorama
import torch


class Slave(object):

	def __init__(self, user: str = None, host: str = "127.0.0.1"):
		self.user = user
		self.host = host

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

	def upload(self, source: str, target: str, recursive: bool =False):
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


def prefix_pipe(pipe_write: int, prefix: str):
	# return a file descriptor for which writes will be prefixed with a string
	prefix_read, prefix_write = os.pipe()

	subprocess.Popen(
		["sed", "-e", f"s/^/{prefix}/;"],
		stdin=prefix_read,
		stdout=pipe_write,
		stderr=None,
	)

	return prefix_write


def print_info(message: str):
	print(colorama.Fore.GREEN + "[localhost|master]:\t" + colorama.Fore.RESET + message)


class Trainer(Slave):

	def __init__(self, config: dict):
		super().__init__(user=config.get("user", None), host=config.get("host", "127.0.0.1"))

		self.executable = config["executable"]
		self.model_path = config["model"]

		if config.get("authorize", False):
			self.authorize()

	def start_training(self, replays_read: int) -> subprocess.Popen:
		command = self.executable
		arguments = [self.model_path]
		prefix_write = prefix_pipe(None, colorama.Fore.CYAN + f"[{self.destination}|training]:\t" + colorama.Fore.RESET)

		return self.command(command, arguments, stdin=replays_read, stderr=prefix_write)

	def download_model(self, target_path: str):
		self.download(self.model_path, target_path)


class Selfplayer(Slave):

	def __init__(self, config: dict):
		super().__init__(user=config.get("user", None), host=config.get("host", "127.0.0.1"))

		self.executable = config["executable"]
		self.model_path = config["model"]

		if config.get("authorize", False):
			self.authorize()

	def start_selfplay(self, replays_write: int) -> subprocess.Popen:
		command = self.executable
		arguments = [self.model_path]
		prefix_write = prefix_pipe(None, colorama.Fore.MAGENTA + f"[{self.destination}|selfplay]:\t" + colorama.Fore.RESET)

		return self.command(command, arguments, stdout=replays_write, stderr=prefix_write)

	def upload_model(self, source_path: str):
		self.upload(source_path, self.model_path)



if __name__ == "__main__":

	colorama.init()

	parser = argparse.ArgumentParser(description="Sigma Zero master script")
	parser.add_argument("config", type=str, help="Path to configuration file")

	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	workdir = config["workdir"]
	model_path = os.path.join(workdir, "model.pt")

	replays_read, replays_write = os.pipe()

	trainers = []
	selfplayers = []

	for training_config in config["training"]:
		trainer = Trainer(training_config)
		process = trainer.start_training(replays_read)
		trainers.append((trainer, process))

		print_info("started trainer")

	for selfplay_config in config["selfplay"]:
		selfplayer = Selfplayer(selfplay_config)
		process = selfplayer.start_selfplay(replays_write)
		selfplayers.append((selfplayer, process))

		print_info("started selfplay")

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
			print_info(f"uploaded model to selfplayer")

	colorama.deinit()
