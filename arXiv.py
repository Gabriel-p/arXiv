
import os
from os.path import join, realpath, dirname
import numpy as np

from bs4 import BeautifulSoup as BS
import requests
import re
from collections import OrderedDict
import shlex
import textwrap
import string
from textblob.classifiers import NaiveBayesClassifier

import nltk.data
from nltk.corpus import stopwords
import pickle


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


def get_arxiv_data(categ):
    '''
    Downloads data from arXiv.
    '''
    print("\nDownloading arXiv data.")
    # url = "http://arxiv.org/list/" + categ + "/new"
    day, month, year = '7', '4', '2017'
    url = "https://arxiv.org/catchup?smonth=" + month + "&group=grp_&s" +\
          "day=" + day + "&num=50&archive=astro-ph&method=with&syear=" + year

    html = requests.get(url)
    soup = BS(html.content, 'lxml')

    # with open("temp", "wb") as f:
    #     f.write(html.content)
    # with open("temp", "rb") as f:
    #     soup = BS(f, 'lxml')

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


def findWholeWord(w, string):
    '''
    Finds a single word or a sequence of words in the list 'string'.
    Ignores upper/lowercasing. Returns True if 'w' was found and
    False if it wasn't.
    '''
    pattern = re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE)
    return True if pattern.search(string) else False


def keywProbability(N_in, N_out):
    """
    """
    if (N_in + N_out) > 0:
        K_p = max(0., (N_in - N_out) / (N_in + N_out))
    else:
        K_p = -1.

    return K_p


def get_Kprob(articles, in_k, ou_k):
    '''
    Obtains keyword base probabilities for each article, according to the
    in/out keywords.
    '''
    art_K_prob = []
    # Loop through each article stored.
    for i, art in enumerate(articles):
        N_in, N_out = 0., 0.
        # Search for rejected words.
        for ou_keyw in ou_k:
            # Search titles, abstract and authors list.
            for words in art[:3]:
                # If the keyword is in any list.
                if findWholeWord(ou_keyw, words):
                    N_out += N_out + 1.
        # Search for accepted keywords.
        for in_keyw in in_k:
            # Search titles, abstract and authors list.
            for words in art[:3]:
                # If the keyword is in any list.
                if findWholeWord(in_keyw, words):
                    N_in += N_in + 1.

        art_K_prob.append(keywProbability(N_in, N_out))

    return art_K_prob


def get_Bprob(mypath, articles):
    '''
    Obtains keyword base probabilities for each article, according to the
    in/out keywords.
    '''
    stpwrds = stopwords.words("english") +\
        ['find', 'data', 'observed', 'using', 'show', 'well',
         'around', 'used', 'thus', 'within', 'investigate', 'also',
         'recently', 'however']
    # Remove punctuation.
    translator = str.maketrans('', '', string.punctuation)

    try:
        with open(join(mypath, "classifier.pkl"), "rb") as f:
            cl = pickle.load(f)

        # abst = []
        art_B_prob = []
        for art in articles:
            title = str(art[1]).lower()
            abstract = str(art[2]).lower().translate(translator)
            # Remove stopwords and some common words while maintaining
            # abstract's order.
            abstract = ' '.join(
                [w for w in list(OrderedDict.fromkeys(abstract.split()))
                 if w not in stpwrds])

            # abst.append(abstract)
            # print("{:.3f}, {:.3f}, {:.3f}".format(
            #     cl.prob_classify(title + ' ' + abstract).prob("neg"),
            #     cl.prob_classify(title + ' ' + abstract).prob("meh"),
            #     cl.prob_classify(title + ' ' + abstract).prob("pos")))

            art_B_prob.append(
                cl.prob_classify(title + ' ' + abstract).prob("pos"))

        # from collections import Counter
        # print(Counter(w for w in ' '.join(abst).split() if len(w) >= 4))

    except FileNotFoundError:
        print("No previous classifier file found.\n")
        art_B_prob, cl = [-1. for _ in articles], []

    return art_B_prob, cl


def sort_rev(articles, K_prob, B_prob):
    '''
    Sort articles according to rank and reverse list so larger values
    will be located first in the list.
    '''
    art_rank = (np.array(K_prob) + np.array(B_prob)) / 2.
    # Sort according to rank values.
    art_zip = list(zip(art_rank, articles, K_prob, B_prob))
    art_zip.sort()
    # Revert so larger values will be first.
    art_s_rev = art_zip[::-1]

    articles, K_prob, B_prob = list(zip(*art_s_rev))[1:]

    return articles, K_prob, B_prob


def main():
    '''
    Query newly added articles to selected arXiv categories, rank them
    according to given keywords, and print out the ranked list.
    '''
    mypath = realpath(join(os.getcwd(), dirname(__file__), 'input'))
    nltk.data.path.append(mypath)

    # Read accepted/rejected keywords and categories from file.
    in_k, ou_k, categs = get_in_out()

    # Get new data from all the selected categories.
    articles = []
    for cat_indx, categ in enumerate(categs):

        # Get data from each category.
        soup = get_arxiv_data(categ)

        # Store titles, links, authors and abstracts into list.
        articles = articles + get_articles(soup)

    print("Obtaining probabilities.")
    # Obtain articles' probabilities according to keywords.
    K_prob = get_Kprob(articles, in_k, ou_k)
    # Obtain articles' probabilities based on Bayesian analysis.
    B_prob, cl = get_Bprob(mypath, articles)
    # Sort articles.
    articles, K_prob, B_prob = sort_rev(articles, K_prob, B_prob)

    train = []
    for i, art in enumerate(articles):
        # Title
        title = str(art[1])
        print('\n' + str(i + 1) + ')', textwrap.fill(title, 77))
        # Authors + arXiv link
        authors = art[0] if len(art[0].split(',')) < 4 else\
            ','.join(art[0].split(',')[:3]) + ', et al.'
        print(textwrap.fill(authors, 77), '\n* ' + str(art[3]) + '\n')
        # Abstract
        abstract = str(art[2])
        print(textwrap.fill(abstract, 80))
        print("\nK_p: {:.2f}, B_p: {:.2f}".format(K_prob[i], B_prob[i]))
        pn = input("B_p (1/2/3): ")
        if pn == '1':
            train.append([title + ' ' + abstract, 'neg'])
        elif pn == '2':
            train.append([title + ' ' + abstract, 'meh'])
        elif pn == '3':
            train.append([title + ' ' + abstract, 'pos'])
        elif pn == "quit":
            break

    if train:
        if cl:
            # Update the classifier with the new training data
            print("\nUpdating classifier.")
            cl.update(train)
        else:
            # Generate classifier
            print("\nGenerating classifier.")
            cl = NaiveBayesClassifier(train)

        with open(join(mypath, "classifier.pkl"), "wb") as f:
            pickle.dump(cl, f)

    print("Finished.")


if __name__ == "__main__":
    main()
