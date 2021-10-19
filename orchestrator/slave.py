"""Slave module."""


import subprocess


class Slave(object):
	
	def __init__(self, host: str = "127.0.0.1", user: str = None):
		self.host = host
		self.user = user
		self.process = None

	@property
	def destination(self):
		if self.user:
			return f"{self.user}@{self.host}"
		else:
			return self.host

	def copy_id(self):
		subprocess.run(
			["ssh-copy-id", self.destination],
			check=True,
			capture_output=True
		)

	def command(self, command: str, arguments: list = None, stdout: int = subprocess.PIPE, stdin: int = subprocess.PIPE, stderr: int = None):
		if not self.process:
			self.process = subprocess.Popen(
				["ssh", self.destination, command] + (arguments if arguments else []),
				stdout=stdout, stdin=stdin, stderr=stderr
			)
		else:
			print("commanding busy slave")

	def is_alive(self) -> bool:
		if self.process:
			return self.process.is_alive()
		else:
			return False

	def terminate(self) -> int:
		status = 0
		
		if self.process:
			self.process.terminate()
			status = self.process.wait()
			self.process = None

		return status
