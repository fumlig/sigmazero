#!/usr/bin/env python3

import os
import argparse
import subprocess
import yaml


class Slave(object):
	
	def __init__(self, host: str = "127.0.0.1", user: str = None):
		self.host = host
		self.user = user

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

	def upload(self, source: str, target: str, recursive=False):
		subprocess.run(
			["scp"] + (["-r"] if recursive else []) + [source, f"{self.destination}:{target}"],
			check=True,
			capture_output=True
		)

	def download(self, source: str, target: str, recursive=False):
		subprocess.run(
			["scp"] + (["-r"] if recursive else []) + [f"{self.destination}:{source}", target],
			check=True,
			capture_output=True
		)

	def authorize(self):
		subprocess.run(
			["ssh-copy-id", self.destination],
			check=True,
			capture_output=True
		)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Sigma Zero master script")
	parser.add_argument("config", type=str, help="Path to configuration file")

	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = yaml.safe_load(f)

	model_path = config["model"]

	replays_read, replays_write = os.pipe()

	trainers = []
	selfplayers = []

	for training_config in config["training"]:
		user = training_config.get("user", None)
		host = training_config.get("host", "127.0.0.1")
		authorize = training_config.get("authorize", False)

		command = training_config["executable"]
		arguments = [training_config["model"]]

		training_slave = Slave(host=host, user=user)

		if authorize:
			training_slave.authorize()

		training_process = training_slave.command(command, arguments, stdin=replays_read)

		trainers.append((training_slave, training_process, training_config))
		

	for selfplay_config in config["selfplay"]:
		user = selfplay_config.get("user", None)
		host = selfplay_config.get("host", "127.0.0.1")
		authorize = selfplay_config.get("authorize", False)

		command = selfplay_config["executable"]
		arguments = [selfplay_config["model"]]

		selfplay_slave = Slave(host=host, user=user)

		if authorize:
			selfplay_slave.authorize()

		selfplay_process = selfplay_slave.command(command, arguments, stdout=replays_write)

		selfplayers.append((selfplay_slave, selfplay_process, selfplay_config))

	while True:
		for (training_slave, training_process, training_config) in trainers:
			while training_event := training_process.stdout.readline().decode().strip():
				# todo: eventually, merging of models can be done here
				
				print("master:", "new model received")

				training_slave.download(training_config["model"], model_path)
				print("master:", "downloaded model from trainer")

				for (selfplay_slave, _, selfplay_config) in selfplayers:
					selfplay_slave.upload(model_path, selfplay_config["model"])
					print("master:", "uploaded model to trainer")