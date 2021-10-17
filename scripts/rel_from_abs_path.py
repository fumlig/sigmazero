#!/usr/bin/env python3

"""Given two absolute paths, get first path relative to second one."""

import os, sys

abs1 = sys.argv[1]
abs2 = sys.argv[2]

common = os.path.commonpath([abs1, abs2])
rel = os.path.relpath(abs1, common)

if os.path.dirname(common) == common:
	# only root in common
	print(abs1)
else:
	print(rel)
