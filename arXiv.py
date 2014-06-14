#!/usr/bin/env python

import urllib
import os
import re
import shlex

"""
Goes through the new submissions to present those more relevant to you.
"""

# Get arXiv/astro-ph/new data.
#file = urllib.urlopen("http://arxiv.org/list/astro-ph/new")
#lines = [str(line) for line in file]
#file.close()

# f = urllib.urlopen("http://arxiv.org/list/astro-ph/new")
# s = f.read()
# f.close()
# ff = open("temp.del", "w")
# ff.write(s)
# ff.close()
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
        article = [title, abstract, ", ".join(authors), link]
        articles.append(article)
        # Reset authors list.
        authors = []

# Read accepted/rejected keywords from file.
in_k, ou_k = [], []
with open("keywords.dat", "r") as ff:
    for li in ff:
        if not li.startswith("#"):
            # Accepted keywords.
            if li[0:2] == 'IN':
                # Store each keyword separately in list.
                for i in shlex.split(li[3:]):
                    in_k.append(i )
            # Rejected keywords.
            if li[0:2] == 'OU':
                ou_k = shlex.split(li[3:])


# Store articles indexes and their ranks.
art_rank = [0.]*len(articles)
# Loop through each article stored.
for art_indx, art in enumerate(articles):
    # Search for rejected words.
    if any(keyword in string for string in art[:3] for keyword in ou_k):
        art_rank[art_indx] = -1.
    else:
        # Search for accepted keywords.
        for in_indx, in_keyw in enumerate(in_k):
            # Search titles, abstract and authors list.
            for lst in art[:3]:
                # If the keyword is in any list.
                if in_keyw in lst:
                    # Assign a value based on its position
                    # in the accepted keywords list (higher
                    # values for earlier keywords)
                    art_rank[art_indx] = art_rank[art_indx] + (1. / (1 + in_indx))

# Sort articles according to its rank values.
art_sorted = [x for (y,x) in sorted(zip(art_rank, articles))]
# Revert so larger values will be first.
art_s_rev = art_sorted[::-1]

for i in range(10):
	print i, art_s_rev[i][0], '\n'



    

