#!/usr/bin/env python

import urllib
import re
import shlex

"""
Goes through the new submissions to present those more relevant to you.
"""


def findWholeWord(w, string):
    '''
    Finds a single word or a sequence of words in the list 'string'.
    Ignores upper/lowercasing. Returns True if 'w' was found and
    False if it wasn't.
    '''
    pattern = re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE)
    return  True if pattern.search(string) else False


def get_arxiv_data(categ):
    '''
    Downloads data from arXiv.
    '''
    ff = urllib.urlopen("http://arxiv.org/list/" + categ + "/new")
    lines = [str(line) for line in ff]
    ff.close()

    return lines


def get_articles():
    '''
    Splits articles into lists containing title, abstract, authors and link.
    Article info is located between <dt> and </dd> tags.
    '''
    articles, authors = [], []
    for index, line in enumerate(lines):
        # Get link.
        if line[0:4] == '<dt>':
            a = re.split('</a>&nbsp;  <span class="list-identifier"><a href="|\
            " title="Abstract">', line)
            b = re.split('" title="Abstract">', a[1])
            link = 'arxiv.org' + b[0]
        # Get title.
        if line[0:38] == '<span class="descriptor">Title:</span>':
            a = re.split('Title:</span> |\n', line)
            title = str(a[1])
        # Get authors.
        if line.startswith('<a href="/find/'):
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
            # Join authors in single list.
            authors_j = ", ".join(authors)
            # Replace \n from abstract with a space.
            abstract_r = abstract.replace("\n", " ")
            article = [authors_j, title, abstract_r, link]
            articles.append(article)
            # Reset authors list.
            authors = []

    return articles


def get_in_out():
    '''
    Reads in/out keywords from file.
    '''
    in_k, ou_k, categs = [], [], []
    with open("keywords.dat", "r") as ff:
        for li in ff:
            if not li.startswith("#"):
                # Categories.
                if li[0:2] == 'CA':
                  # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        categs.append(i)
                # Accepted keywords.
                if li[0:2] == 'IN':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        in_k.append(i)
                # Rejected keywords.
                if li[0:2] == 'OU':
                  # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        ou_k.append(i)

    return in_k, ou_k, categs


def get_rank(articles, in_k, ou_k):
    '''
    Ranks articles according to the in keywords and rejects those that
    contain any out keywords.
    '''
    art_rank = [0.] * len(articles)
    # Loop through each article stored.
    for art_indx, art in enumerate(articles):
        # Search for rejected words.
            for ou_indx, ou_keyw in enumerate(ou_k):
                # Search titles, abstract and authors list.
                for ata_indx, lst in enumerate(art[:3]):
                    # If the keyword is in any list.
                    if findWholeWord(ou_keyw, lst):
                        # Assign a value based on its position
                        # in the rejected keywords list (higher
                        # values for earlier keywords)
                        art_rank[art_indx] = art_rank[art_indx] - \
                        ((3. - ata_indx) / (1. + ou_indx))
            # Search for accepted keywords.
            for in_indx, in_keyw in enumerate(in_k):
                # Search titles, abstract and authors list.
                for ata_indx, lst in enumerate(art[:3]):
                    # If the keyword is in any list.
                    if findWholeWord(in_keyw, lst):
                        # Assign a value based on its position
                        # in the accepted keywords list (higher
                        # values for earlier keywords)
                        art_rank[art_indx] = art_rank[art_indx] + \
                        ((3. - ata_indx) / (1. + in_indx))
    return art_rank


def sort_rev(art_rank, articles):
    '''
    Sort articles according to rank and reverse list so larger values
    will be located first in the list.
    '''
    # Sort according to rank.
    art_sorted = [x for (y, x) in sorted(zip(art_rank, articles))]
    # Revert so larger values will be first.
    art_s_rev = art_sorted[::-1]

    return art_s_rev


# Read accepted/rejected keywords and categories from file.
in_k, ou_k, categs = get_in_out()

# Get new data from all the selected categories.
articles = []
for cat_indx, categ in enumerate(categs):

    # Get data from each category.
    lines = get_arxiv_data(categ)

    # Store titles, links, authors and abstracts into list.
    articles = articles + get_articles()

# Obtain articles' ranks according to keywords.
art_rank = get_rank(articles, in_k, ou_k)

# Sort articles according to rank values.
art_s_rev = sort_rev(art_rank, articles)

for i in range(15):
    print i, art_s_rev[i][1], ' (', art_s_rev[i][3], ')'
    print art_s_rev[i][0], '\n'
    print art_s_rev[i][2], '\n'
