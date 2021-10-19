#!/usr/bin/env python3

import os
from slave import Slave

test = Slave()

replays_read, replays_write = os.pipe()

test.command("./studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/selfplay", stdout=replays_write)

with open(replays_read, 'r') as replays_file:
	while True:
		print(replays_file.readline())