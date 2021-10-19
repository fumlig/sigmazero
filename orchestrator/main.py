#!/usr/bin/env python3

import os
from slave import Slave

selfplay = Slave()
training = Slave()

replays_read, replays_write = os.pipe()

selfplay.copy_id()
training.copy_id()

selfplay.command("./studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/selfplay", stdout=replays_write)
training.command("./studies/tdde19-advanced-project-course-ai-and-machine-learning/tjack/build/training", stdin=replays_read)

while True:
	print(training.process.stdout.readline())