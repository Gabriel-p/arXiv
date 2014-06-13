#!/usr/bin/env python

import urllib
import os

"""
Goes through the new submissions to present those more relevant to you.
"""

# Get arXiv/astro-ph/new data.
# file = urllib.urlopen("http://arxiv.org/list/astro-ph/new")
# lines = [str(line) for line in file]
# file.close()

#f = urllib.urlopen("http://arxiv.org/list/astro-ph/new")
#s = f.read()
#f.close()
# Write data to temp file.
#ff = open("temp.del", "w")
#ff.write(s)
#ff.close()
lines = open("temp.del", "r")

# Abstracts are located between <dt> and </dd>

# Find number of bugs left.
ini_block, end_block = [], []
for index, line in enumerate(lines):
	#print index, line[0:4]
	if line[0:4] == '<dt>':
		a = line.split(']</a>&nbsp;')
		b = a[0].split('">[')
		art_num = int(b[1])
		ini_block.append(index)
		raw_input()
	if line == '</dd>':
		end_block.append(index)

	

