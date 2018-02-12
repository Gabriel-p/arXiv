
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
import requests
import shlex
import textwrap
from datetime import date, timedelta


def main():
    '''
    Query newly added articles to selected arXiv categories, rank them,
    print the ranked list, and ask for manual ranking.
    '''
    # Read date range, arXiv categories, and classification mode.
    mode, date_range, categs, clmode = get_in_out()

    # Download articles from arXiv.
    articles, dates = downArts(mode, date_range, categs)

    # Read previous classifications.
    wordsRank = readWords()

    # Obtain articles' probabilities based on Bayesian analysis.
    ranks, B_prob = get_Bprob(clmode, wordsRank, articles)

    # Sort articles.
    articles, ranks, B_prob = sort_rev(articles, ranks, B_prob)

    # Manual ranking.
    train = manualRank(articles, dates, ranks, B_prob)

    # Update classifier data.
    updtRank(wordsRank, train)

    print("\nFinished.")


def get_in_out():
    '''
    Reads input parameters from file.
    '''
    categs = []
    with open("in_params.dat", "r") as ff:
        for li in ff:
            if not li.startswith("#"):
                # Mode.
                if li[0:2] == 'MO':
                    # Store each keyword separately in list.
                    mode, start_date, end_date = shlex.split(li[3:])
                # Categories.
                if li[0:2] == 'CA':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        categs.append(i)
                # Classification mode.
                if li[0:2] == 'CM':
                    clmode = li.split()[1]

    return mode, [start_date, end_date], categs, clmode


def dateRange(date_range):
    """
    Store individual dates for a range, skipping weekends.
    """
    start_date = list(map(int, date_range[0].split('-')))
    end_date = list(map(int, date_range[1].split('-')))

    ini_date, end_date = date(*start_date), date(*end_date)

    d, delta, weekend = ini_date, timedelta(days=1), [5, 6]
    dates_no_wknds = []
    while d <= end_date:
        if d.weekday() not in weekend:
            # Store as [year, month, day]
            dates_no_wknds.append(str(d).split('-'))
        d += delta

    return dates_no_wknds


def get_arxiv_data(categ, day_week):
    '''
    Downloads data from arXiv.
    '''
    if day_week == '':
        print("Downloading latest arXiv data.")
        url = "http://arxiv.org/list/" + categ + "/new"
    else:
        year, month, day = day_week
        print("Downloading arXiv data for {}-{}-{}.".format(
            year, month, day))
        url = "https://arxiv.org/catchup?smonth=" + month + "&group=grp_&s" +\
              "day=" + day + "&num=50&archive=astro-ph&method=with&syear=" +\
              year

    html = requests.get(url)
    soup = BS(html.content, 'lxml')

    # with open("temp", "wb") as f:
    #     f.write(html.content)
    # with open("temp", "rb") as f:
    #     soup = BS(f, 'lxml')

    return soup


def downArts(mode, date_range, categs):
    """
    Download articles from arXiv for all the categories selected, for the
    dates chosen.
    """
    dates_no_wknds = ['']
    if mode == 'range':
        dates_no_wknds = dateRange(date_range)

    # Download articles from arXiv.
    articles, dates = [], []
    for day_week in dates_no_wknds:
        # Get new data from all the selected categories.
        for cat_indx, categ in enumerate(categs):

            # Get data from each category.
            soup = get_arxiv_data(categ, day_week)

            # Store titles, links, authors and abstracts into list.
            date_arts = get_articles(soup)
            articles = articles + date_arts

            # Dates
            dates = dates + ['-'.join(day_week) for _ in date_arts]

    # import pickle
    # # with open('filename.pkl', 'wb') as f:
    # #     pickle.dump((articles, dates), f)
    # with open('filename.pkl', 'rb') as f:
    #     articles, dates = pickle.load(f)

    return articles, dates


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


def readWords():
    """
    Read ranked words from input file.
    """
    print("\nRead previous classification.")
    try:
        wordsRank = pd.read_csv(
            "classifier.dat", header=None, names=("date", "rank", "articles"))
    except FileNotFoundError:
        wordsRank = pd.DataFrame([])

    return wordsRank


def get_Bprob(clmode, wordsRank, articles):
    '''
    Obtains Bayesian probabilities for each article.

    Based on the example: http://scikit-learn.org/stable/tutorial/
    text_analytics/working_with_text_data.html

    Uses the classes:
    - sklearn.naive_bayes.MultinomialNB
    - sklearn.linear_model.SGDClassifier
    '''
    if not wordsRank.empty:
        modes = {
            'NB': 'Naive Bayes', 'LR': 'Logistic regression',
            'MH': 'Modified Huber', 'SVM': 'Support Vector Machines',
            'PC': 'Perceptron'}
        print("Training classifier ({}).".format(modes[clmode]))

        # Extract titles and abstracts.
        titles, abstracts = list(zip(*articles))[1], list(zip(*articles))[2]
        titlAbs = [_ + ' ' + abstracts[i] for i, _ in enumerate(titles)]

        # This block is the same as the Pipeline() below, but done in parts.
        # # Tokenizing text
        # count_vect = CountVectorizer()
        # X_train_counts = count_vect.fit_transform(wordsRank['articles'])
        # # From occurrences to frequencies
        # tfidf_transformer = TfidfTransformer()
        # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # # Training a classifier
        # clf = MultinomialNB().fit(X_train_tfidf, wordsRank['rank'])
        # # Predict based on titles and abstract data.
        # X_new_counts = count_vect.transform(titlAbs)
        # X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        # B_prob = clf.predict(X_new_tfidf)

        if clmode == 'NB':
            # NaiveBayes
            text_clf = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                 ('clf', MultinomialNB())])
        elif clmode == 'LR':
            # Logistic regression
            text_clf = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(loss='log', max_iter=1000, tol=1e-3))])
        elif clmode == 'MH':
            # Modified Huber
            text_clf = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(
                     loss='modified_huber', max_iter=1000, tol=1e-3))])
        elif clmode == 'SVM':
            # Support Vector Machines
            text_clf = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(max_iter=1000, tol=1e-3))])
        elif clmode == 'PC':
            # Perceptron
            text_clf = Pipeline(
                [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                 ('clf', SGDClassifier(
                     loss='perceptron', max_iter=1000, tol=1e-3))])

        # Train the model.
        text_clf.fit(wordsRank['articles'], wordsRank['rank'])
        # Predict classifications.
        ranks = text_clf.predict(titlAbs)

        # Probability estimates are only available for 'log' and
        # 'modified Huber' loss when using the 'SGDClassifier()'.
        if clmode not in ['SVM', 'PC']:
            B_prob = text_clf.predict_proba(titlAbs).max(axis=1)
        else:
            print("  WARNING: probability estimates are not available"
                  "for this method.")
            B_prob = np.ones(len(articles))

    else:
        ranks, B_prob = [0 for _ in articles], [0 for _ in articles]
        print("No previous classifier file found.")

    return ranks, B_prob


def sort_rev(articles, ranks, B_prob):
    '''
    Sort articles according to rank values first and probabilities second
    and reverse list so larger values  will be located first in the list.
    '''
    # Sort.
    ranks, B_prob, articles = (
        list(t) for t in zip(*sorted(zip(ranks, B_prob, articles))))
    # Revert.
    articles, ranks, B_prob = articles[::-1], ranks[::-1], B_prob[::-1]

    return articles, ranks, B_prob


def manualRank(articles, dates, ranks, B_prob):
    """
    Manually rank articles.

    'q', 'quit', 'quit()', 'exit' exit the ranking process and stores
    whatever was ranked at that point.
    """
    print("Articles to classify: {}".format(len(articles)))
    train = []
    for i, art in enumerate(articles):
        print('\n{}) R={:.0f}, P={:.2f}, ({})\n'.format(
            str(i + 1), ranks[i], B_prob[i], art[3]))
        # Authors
        authors = art[0] if len(art[0].split(',')) < 4 else\
            ','.join(art[0].split(',')[:3]) + ', et al.'
        print(textwrap.fill(authors, 77) + '\n')
        # Title
        print(textwrap.fill(art[1], 75) + '\n')
        # Abstract
        print(textwrap.fill(art[2], 80))

        # Manual ranking.
        while True:
            pn = input("Rank (1, 2, 3): ")
            # import random
            # pn = random.choice(['1', '2', '3'])
            if pn in ['1', '2', '3']:
                train.append([dates[i], int(pn), art[1] + ' ' + art[2]])
                break
            elif pn == '':
                break
            elif pn in ['q', 'quit', 'quit()', 'exit']:
                return train

    return train


def updtRank(wordsRank, train):
    """
    Update the ranked words file.
    """
    print("\nStoring new classified articles.")
    train = pd.DataFrame(train, columns=("date", "rank", "articles"))
    df = wordsRank.append(train, ignore_index=True)
    df_sort = df.sort_values('rank', ascending=False)
    df_sort.to_csv("classifier.dat", index=False, header=False)


if __name__ == "__main__":
    main()
