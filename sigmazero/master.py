#!/usr/bin/env python3

import os
import subprocess


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


TRAINING = "~/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/training"
SELFPLAY = "~/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/selfplay"
MODEL = "~/studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/model.pt"

if __name__ == "__main__":
	replays_read, replays_write = os.pipe()

	trainer = Slave()
	trainer.authorize()
	training_process = trainer.command(TRAINING, [MODEL], stdin=replays_read)

	selfplayers = [Slave() for _ in range(2)]
	for selfplayer in selfplayers:
		selfplayer.authorize()
		selfplayer.command(SELFPLAY, [MODEL], stdout=replays_write)

	while True:
		print(training_process.stdout.readline().decode().strip())
