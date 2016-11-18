#!/usr/bin/env python

from bs4 import BeautifulSoup as BS
from urllib.request import urlopen
import re
import shlex
import textwrap


def findWholeWord(w, string):
    '''
    Finds a single word or a sequence of words in the list 'string'.
    Ignores upper/lowercasing. Returns True if 'w' was found and
    False if it wasn't.
    '''
    pattern = re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE)
    return True if pattern.search(string) else False


def get_arxiv_data(categ):
    '''
    Downloads data from arXiv.
    '''
    html = urlopen("http://arxiv.org/list/" + categ + "/new")
    soup = BS(html, 'lxml')

    return soup


def get_articles(soup):
    '''
    Splits articles into lists containing title, abstract, authors and link.
    Article info is located between <dt> and </dd> tags.
    '''
    # Get links.
    links = ['https://' + _.text.split()[0].replace(':', '.org/abs/') for _ in
             soup.find_all(class_="list-identifier")]
    # Get titles.
    titles = [_.text.replace('\n', '').replace('Title: ', '') for _ in
              soup.find_all(class_="list-title mathjax")]
    # Get authors.
    authors = [_.text.replace('\n', '').replace('Authors:', '')
               for _ in soup.find_all(class_="list-authors")]
    # Get abstract.
    abstracts = [_.text.replace('\n', ' ') for _
                 in soup.find_all('p', class_="mathjax")]

    articles = list(zip(*[authors, titles, abstracts, links]))

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


def main(N_art):
    '''
    Query newly added articles to selected arXiv categories, rank them
    according to given keywords, and print out the ranked list.
    '''
    # Read accepted/rejected keywords and categories from file.
    in_k, ou_k, categs = get_in_out()

    # Get new data from all the selected categories.
    articles = []
    for cat_indx, categ in enumerate(categs):

        # Get data from each category.
        soup = get_arxiv_data(categ)

        # Store titles, links, authors and abstracts into list.
        articles = articles + get_articles(soup)

    # Obtain articles' ranks according to keywords.
    art_rank = get_rank(articles, in_k, ou_k)

    # Sort articles according to rank values.
    art_s_rev = sort_rev(art_rank, articles)

    print('\n\n')
    for i in range(N_art):
        # Title
        title = str(art_s_rev[i][1])
        print(str(i + 1) + ')', textwrap.fill(title, 77))
        # Authors + arXiv link
        authors = art_s_rev[i][0]
        print(textwrap.fill(authors, 77), '(' + str(art_s_rev[i][3]) + ')\n')
        # Abstract
        abstract = str(art_s_rev[i][2])
        print(textwrap.fill(abstract, 80), '\n\n')


if __name__ == "__main__":

    N_art = input('Number of articles to list (N): ')
    try:
        N_art = int(N_art)
    except:
        N_art = 10
        print('Error. Using default value (N = {}).'.format(N_art))

    main(N_art)
