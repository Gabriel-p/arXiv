#!/usr/bin/env python

import urllib
import os
import re

"""
Goes through the new submissions to present those more relevant to you.
"""

# Get arXiv/astro-ph/new data.
#file = urllib.urlopen("http://arxiv.org/list/astro-ph/new")
#lines = [str(line) for line in file]
#file.close()
lines = [str(line) for line in open("temp.del", "r")]

# Article info is located between <dt> and </dd> tags.

# Store titles, links, authors and abstracts.
articles, authors = [], []
for index, line in enumerate(lines):
	# Get link.
	if line[0:4] == '<dt>':
		a = re.split('</a>&nbsp;  <span class="list-identifier"><a href="|" title="Abstract">', line)
		link = 'arxiv.orx' + a[1]
	# Get title.
	if line[0:38] == '<span class="descriptor">Title:</span>':
		a = re.split('Title:</span> |\n', line)
		title = str(a[1])
	# Get authors.
	if line[0:28] == '<a href="/find/astro-ph/1/au':
		a = re.split('all/0/1">|</a>', line)
		authors.append(str(a[1]))
	# Get abstract.
	if line[0:3] == '<p>':
		abstract = re.split('<p>', line)[1]
		i = 1
		while lines[index + i] != '</p>\n':
			abstract = abstract + str(lines[index + i])
			i += 1
	# End of article.
	if line == '</dd>\n':
		# Store in single list.
		article = [title, abstract, authors, link]
		articles.append(article)
		# Reset authors list.
		authors = []

# Read accepted/rejected keywords from file.
with open("keywords.day", "r") as ff:
	for li in ff:
		if not li.startswith("#"):
			if li[0:2] == 'IN':
				in_k = [li[3:].split()]

print in_k


	

