
from configparser import ConfigParser
import numpy as np
import pandas as pd
from shutil import copyfile
from datetime import date, timedelta
from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
from .process import arxivDwnld, articlList, filterSubcats


def readINI():
    """
    Reads input parameters from .ini file.
    """
    pars = ConfigParser()
    pars.read('params.ini')

    mode = pars['Data input']['mode']
    start_date = pars['Data input']['start']
    end_date = pars['Data input']['end']

    categs = pars['Categories']['cat'].split()
    subcategs = pars['Categories']['subcat'].split()
    clmode = pars['Classification']['mode']
    groups = pars['Interest groups']['names'].split()

    # Add 'not_interested' group at the end.
    groups.append('not_interested')

    # Ids of all defined groups starting from 1, as strings.
    gr_ids = [str(_) for _ in range(1, len(groups))] + ['n']

    print("\nRunning '{}' mode.".format(mode))
    print("Classifier '{}' selected.\n".format(clmode))
    return mode, [start_date, end_date], categs, subcategs, groups, gr_ids,\
        clmode


def downArts(mode, date_range, categs, subcategs):
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

    if mode == 'zotero':
        articles, dates = readZotero()
    else:
        # Download articles from arXiv.
        articles, dates = [], []
        for day_week in dates_no_wknds:
            # Get new data from all the selected categories.
            for cat_indx, categ in enumerate(categs):

                # Get data from each category.
                soup = arxivDwnld(categ, day_week, mode)

                # Store article data list.
                soup_arts = articlList(soup)

                # Filter by sub-categories
                date_arts = filterSubcats(subcategs, soup_arts)

                # Filter out duplicated articles.
                if articles:
                    all_arts = list(zip(*articles))
                    no_dupl = []
                    for art in date_arts:
                        # Compare titles.
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
    init_date = date(2006, 1, 2)  # date(1995, 1, 1)

    all_days = rrule(
        DAILY, dtstart=init_date, until=date.today(),
        byweekday=(MO, TU, WE, TH, FR))
    N_days = all_days.count()
    r_idx = np.random.choice(range(N_days))
    rand_date = [str(all_days[r_idx].date()).split('-')]

    return rand_date


def readWords():
    """
    Read ranked articles from the input classification file.
    """
    print("\nRead previous classification.")
    try:
        wordsRank = pd.read_csv(
            "classifier.dat", header=None, names=("date", "rank", "articles"))
    except FileNotFoundError:
        wordsRank = pd.DataFrame([])

    return wordsRank


def readZotero():
    """
    Read a Zotero .csv file generated using: 'Export collection' with
    'Format: CSV' not exporting notes.
    """
    print("Read Zotero CSV file.")
    zot = pd.read_csv(
        "zotero.csv",
        usecols=('Author', 'Title', 'Abstract Note', 'Url', 'Date'))
    N_orig = len(zot)

    # Drop al 'Nan' values that might be present.
    zot = zot.dropna(axis=0, how='any')
    # Drop articles with no abstracts available.
    zot = zot[zot['Abstract Note'] != "Not Available"]

    # Re order, drop dates.
    art_data = zot[['Author', 'Title', 'Abstract Note', 'Url']]

    # Pack in required format.
    articles = list(zip(*[list(art_data[_]) for _ in art_data]))
    dates = list(zot['Date'])

    print("{} articles read, {} have all the needed data.".format(
        N_orig, len(art_data)))

    return articles, dates


def updtRank(wordsRank, train):
    """
    Update the ranked words file.
    """
    if train:

        # Back up classifier file.
        copyfile("classifier.dat", "classifier_bck.dat")

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
