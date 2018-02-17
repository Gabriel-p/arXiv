
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
from itertools import groupby
from datetime import date, timedelta
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR


def main():
    '''
    Query newly added articles to selected arXiv categories, rank them,
    print the ranked list, and ask for manual ranking.

    Ranking:

    1: Primary interest.
    2: Secondary interest.
    3: Tertiary interest.
    4: ....

    '''

    # Read date range, arXiv categories, and classification mode.
    mode, date_range, categs, groups, clmode = get_in_out()

    # Download articles from arXiv.
    articles, dates = downArts(mode, date_range, categs)

    if articles:
        # Read previous classifications.
        wordsRank = readWords()

        # Obtain articles' probabilities based on ML analysis.
        ranks, probs = get_Bprob(clmode, wordsRank, articles)

        # Sort and group articles.
        grpd_arts = sort_rev(articles, dates, ranks, probs)

        # Manual ranking.
        train = manualRank(groups, grpd_arts)

        # Update classifier data.
        updtRank(wordsRank, train)
    else:
        print("No articles found for the date(s) selected.")

    print("\nFinished.")


def get_in_out():
    '''
    Reads input parameters from file.
    '''
    categs, groups = [], []
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
                if li[0:2] == 'IG':
                    # Store each keyword separately in list.
                    for i in shlex.split(li[3:]):
                        groups.append(i)

    print("\nRunning '{}' mode.".format(mode))
    print("Classifier '{}' selected.\n".format(clmode))
    return mode, [start_date, end_date], categs, groups, clmode


def dateRange(date_range):
    """
    Store individual dates for a range, skipping weekends.
    """
    start_date = list(map(int, date_range[0].split('-')))
    end_date = list(map(int, date_range[1].split('-')))

    ini_date, end_date = date(*start_date), date(*end_date)
    if ini_date > end_date:
        raise ValueError("The 'end_date' must be larger than 'start_date'.")

    d, delta, weekend = ini_date, timedelta(days=1), [5, 6]
    dates_no_wknds = []
    while d <= end_date:
        if d.weekday() not in weekend:
            # Store as [year, month, day]
            dates_no_wknds.append(str(d).split('-'))
        d += delta

    return dates_no_wknds


def dateRandom():
    """
    Select random date, skipping weekends.
    """

    all_days = rrule(
        DAILY, dtstart=date(1995, 1, 1), until=date.today(),
        byweekday=(MO, TU, WE, TH, FR))
    N_days = all_days.count()
    r_idx = np.random.choice(range(N_days))
    rand_date = [str(all_days[r_idx].date()).split('-')]

    return rand_date


def get_arxiv_data(categ, day_week, mode):
    '''
    Downloads data from arXiv.
    '''
    if mode == 'recent':
        print("Downloading latest arXiv data.")
        url = "http://arxiv.org/list/" + categ + "/new"
    else:
        year, month, day = day_week
        print("Downloading arXiv data for {}-{}-{}".format(
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
    if mode == 'recent':
        dates_no_wknds = [str(date.today()).split('-')]
    elif mode == 'range':
        dates_no_wknds = dateRange(date_range)
    elif mode == 'random':
        dates_no_wknds = dateRandom()

    # Download articles from arXiv.
    articles, dates = [], []
    for day_week in dates_no_wknds:
        # Get new data from all the selected categories.
        for cat_indx, categ in enumerate(categs):

            # Get data from each category.
            soup = get_arxiv_data(categ, day_week, mode)

            # Store titles, links, authors and abstracts into list.
            date_arts = get_articles(soup)

            # Filter out duplicated articles.
            if articles:
                all_arts = list(zip(*articles))
                no_dupl = []
                for art in date_arts:
                    if art[1] not in all_arts[1]:
                        no_dupl.append(art)
            else:
                no_dupl = date_arts
            # Add unique elements to list.
            articles = articles + no_dupl

            # Dates
            dates = dates + ['-'.join(day_week) for _ in no_dupl]

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
    Read ranked articles from input classification file.
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
    Rank and obtains probabilities for each article.

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
        # probs = clf.predict(X_new_tfidf)

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
            probs = text_clf.predict_proba(titlAbs).max(axis=1)
        else:
            print("  WARNING: probability estimates are not available"
                  "for this method.")
            probs = np.ones(len(articles))

    else:
        ranks, probs = [0 for _ in articles], [0. for _ in articles]
        print("No previous classifier file found.")

    return ranks, probs


def sort_rev(articles, dates, ranks, probs):
    '''
    Sort articles according to rank values first and probabilities second
    in reverse order so larger probabilities will be positioned first.

    Group articles in groups by ranks.
    '''
    # Sort.
    ranks, probs, articles, dates = (
        list(t) for t in zip(*sorted(zip(
            ranks, -np.array(probs), articles, dates))))
    # Revert back.
    probs = -np.array(probs)

    # Group by ranks.
    data = list(zip(*[ranks, probs, articles, dates]))
    c = groupby(data, lambda x: x[0])
    grpd_arts = {}
    for k, group in c:
        grpd_arts[k] = list(group)
    # To list and sublists.
    grpd_arts = [list(_) for _ in grpd_arts.values()]

    return grpd_arts


def artClass(gr_ids, rank):
    """
    Manual ranking.
    """
    while True:
        pn = input("Rank (1,..,{}): ".format(len(gr_ids)))
        # import random
        # pn = random.choice(['1', '2', '3'])
        if pn in gr_ids:
            return pn
        # Jump to next group.
        elif pn == 'ng':
            if str(rank + 1) in gr_ids:
                print("\nJump to next group.")
                return 'next_gr'
            else:
                print(" No such group '{}' exists.".format(rank + 1))
        # Empty string means don't classify this article and move forward.
        # Rest of options mean "Quit training/classifying."
        elif pn in ['', 'q', 'quit', 'quit()', 'exit']:
            return pn


def manualRank(groups, grpd_arts):
    """
    Manually rank articles.

    'q', 'quit', 'quit()', 'exit' exit the ranking process and stores
    whatever was ranked at that point.
    """
    # Ids of all defined groups starting from 1, as strings.
    gr_ids = [str(_) for _ in range(1, len(groups) + 1)]
    # Total number of articles.
    N_arts = sum([len(_) for _ in grpd_arts])

    print("\nTotal number of articles to classify: {}".format(N_arts))
    print("\nArticles per group defined:")
    for i, g in enumerate(groups):
        print(" {} ({}): {}".format(i + 1, g, len(grpd_arts[i])))

    train = []
    # For each defined group.
    for i, articles_gr in enumerate(grpd_arts):
        print("\n* Articles classified in group {} ({})".format(
            i + 1, groups[i]))

        # For each article in this group.
        for j, data in enumerate(articles_gr):
            rank, prob, art, date = data

            print('\n{}) R={:.0f}, P={:.2f}, {} ({})\n'.format(
                str(j + 1), rank, prob, date, art[3]))
            # Authors
            authors = art[0] if len(art[0].split(',')) < 4 else\
                ','.join(art[0].split(',')[:3]) + ', et al.'
            print(textwrap.fill(authors, 77) + '\n')
            # Title
            print(textwrap.fill(art[1], 75) + '\n')
            # Abstract
            print(textwrap.fill(art[2], 80))

            answ = artClass(gr_ids, rank)

            if answ in gr_ids:
                train.append([date, int(answ), art[1] + ' ' + art[2]])
            # Jump to next group.
            elif answ == 'next_gr':
                break
            # Don't classify this article and move forward.
            elif answ == '':
                pass
            # Quit training/classifying.
            elif answ in ['q', 'quit', 'quit()', 'exit']:
                return train

    return train


def updtRank(wordsRank, train):
    """
    Update the ranked words file.
    """
    if train:
        print("\nStoring {} new classified articles.".format(len(train)))
        train = pd.DataFrame(train, columns=("date", "rank", "articles"))
        # Append existing classification with new one.
        df = wordsRank.append(train, ignore_index=True)
        # Sort according to groups (rank)
        df_sort = df.sort_values('rank', ascending=True)
        # Remove duplicated entried with matchin ranks and data.
        df_unq = df_sort.drop_duplicates(subset=['rank', 'articles'])
        if len(df_unq) < len(df_sort):
            print("Dropping {} duplicated articles.".format(
                len(df_sort) - len(df_unq)))
        # Write to article.
        df_unq.to_csv("classifier.dat", index=False, header=False)


if __name__ == "__main__":
    main()
