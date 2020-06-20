
import numpy as np
import textwrap
from itertools import groupby
from operator import itemgetter

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


def probs(clmode, wordsRank, articles):
    """
    Rank and obtains probabilities for each article.

    Based on the example: http://scikit-learn.org/stable/tutorial/
    text_analytics/working_with_text_data.html
    """

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
        # (https://scikit-learn.org/stable/modules/generated/
        # sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.
        # SGDClassifier.predict_proba)
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


def artSort(gr_ids, articles, dates, ranks, probs):
    """
    Sort articles according to rank values first and probabilities second
    in reverse order so larger probabilities will be positioned first.

    Group articles by ranks.
    """
    # Sort.
    ranks, probs, articles, dates = (
        list(t) for t in zip(*sorted(zip(
            ranks, -np.array(probs), articles, dates))))
    # Revert back.
    probs = -np.array(probs)

    # Group by ranks.
    data = list(zip(*[ranks, probs, articles, dates]))
    gr_data = groupby(data, lambda x: x[0])
    grpd_arts = {}
    for k, group in gr_data:
        grpd_arts[k] = list(group)

    # Store number of articles for each group.
    gr_len = []
    for g_id in gr_ids:
        g_id = int(g_id) if g_id != 'n' else 999
        try:
            gr_len.append(len(grpd_arts[g_id]))
        except KeyError:
            gr_len.append(0)

    # To list and sublists.
    grpd_arts = [list(_) for _ in grpd_arts.values()]

    # Reverse last group (not_interested) so those that have a *lower*
    # probability of belonging here are shown first.
    grpd_arts[-1].sort(key=itemgetter(1))

    return grpd_arts, gr_len


def manual(groups, gr_ids, grpd_arts, gr_len):
    """
    Manually rank articles.

    'q', 'quit', 'quit()', 'exit' exit the ranking process and stores
    whatever was ranked at that point.
    """
    # Total number of articles.
    N_arts = sum([len(_) for _ in grpd_arts])

    print("\nTotal number of articles to classify: {}".format(N_arts))
    print("\nArticles per group defined:")
    for i, g in enumerate(groups):
        print(" G={} ({}): {}".format(gr_ids[i], g, gr_len[i]))
    print("\nInput 'ng' to stop classifying and jump to the next group.")

    train = []
    # For each defined group.
    for articles_gr in grpd_arts:
        gr_id = articles_gr[0][0] - 1 if articles_gr[0][0] != 999 else -1
        print("\n* Articles classified in group '{}' ({})".format(
            groups[gr_id], len(articles_gr)))

        # For each article in this group.
        for j, data in enumerate(articles_gr):
            rank, prob, art, date = data

            r = str(rank) if rank != 999 else 'n'
            print('\n{}/{}) G={}, P={:.2f}, {} ({})\n'.format(
                str(j + 1), len(articles_gr), r, prob, date, art[3]))
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
                a = int(answ) if answ != 'n' else 999
                train.append([date, a, art[1] + ' ' + art[2]])
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


def artClass(gr_ids, rank):
    """
    Manual ranking.
    """
    while True:
        pn = input("Group (1,..,{},n): ".format(len(gr_ids) - 1))
        if pn in gr_ids:
            return pn
        # Jump to next group.
        elif pn == 'ng':
            if str(rank + 1) in gr_ids or (rank + 1) == len(gr_ids):
                print("\nJump to next group.")
                return 'next_gr'
            else:
                print(" No such group '{}' exists.".format(rank + 1))
        # Empty string means don't classify this article and move forward.
        # Rest of options mean "Quit training/classifying."
        elif pn in ['', 'q', 'quit', 'quit()', 'exit']:
            return pn


def zotero(groups, gr_ids, articles, dates):
    """
    Assign a rank to Zotero articles.
    """
    print("\nGroups defined:")
    for i, g in enumerate(groups):
        print(" {}: {}".format(gr_ids[i], g))

    print("\nAssign a group for Zotero entries read.")
    while True:
        pn = input("Group (1,..,{},n): ".format(len(gr_ids) - 1))
        if pn in gr_ids:
            break

    train = []
    for i, art in enumerate(articles):
        train.append([dates[i], int(pn), art[1] + ' ' + art[2]])

    return train
