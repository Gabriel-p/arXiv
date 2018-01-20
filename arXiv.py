
from bs4 import BeautifulSoup as BS
import requests
import shlex
import textwrap
import string
from datetime import date, timedelta


def get_in_out():
    '''
    Reads in/out keywords from file.
    '''
    in_k, ou_k, categs = [], [], []
    with open("keywords.dat", "r") as ff:
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

    return mode, [start_date, end_date], in_k, ou_k, categs


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

    return ini_date, end_date, dates_no_wknds


def get_arxiv_data(categ, day_week):
    '''
    Downloads data from arXiv.
    '''
    if day_week == '':
        url = "http://arxiv.org/list/" + categ + "/new"
    else:
        year, month, day = day_week
        url = "https://arxiv.org/catchup?smonth=" + month + "&group=grp_&s" +\
              "day=" + day + "&num=50&archive=astro-ph&method=with&syear=" +\
              year

    html = requests.get(url)
    soup = BS(html.content, 'lxml')

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


def cleanText(text):
    """
    Remove punctuation and change to lowercase.
    """
    translator = str.maketrans('', '', string.punctuation)
    clean = str(text).lower().translate(translator)

    return clean


def keywProbability(N_in, N_out):
    """
    """
    if (N_in + N_out) > 0:
        K_p = max(0., (2. * N_in - N_out))
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
            for strng in art[:3]:
                N_out = N_out + cleanText(strng).count(ou_keyw.lower())
        # Search for accepted keywords.
        for in_keyw in in_k:
            # Search titles, abstract and authors list.
            for strng in art[:3]:
                N_in = N_in + cleanText(strng).count(in_keyw.lower())

        art_K_prob.append(keywProbability(N_in, N_out))

    return art_K_prob


def sort_rev(articles, K_prob):
    '''
    Sort articles according to rank so larger values will be located last.
    '''
    articles = [x for _, x in sorted(zip(K_prob, articles))]

    return articles, sorted(K_prob)


def main():
    '''
    Query newly added articles to selected arXiv categories, rank them
    according to given keywords, and print out the ranked list.
    '''
    # Read accepted/rejected keywords and categories from file.
    mode, date_range, in_k, ou_k, categs = get_in_out()

    dates_no_wknds = ['']
    if mode == 'range':
        start_date, end_date, dates_no_wknds = dateRange(date_range)
        print("\nDownloading arXiv data for range {} / {}".format(
            start_date, end_date))
    elif mode == 'recent':
        print("\nDownloading recent arXiv data.")
    else:
        print("Unknown mode {}".format())
        raise ValueError

    for day_week in dates_no_wknds:

        # Get new data from all the selected categories.
        articles = []
        for cat_indx, categ in enumerate(categs):

            # Get data from each category.
            soup = get_arxiv_data(categ, day_week)

            # Store titles, links, authors and abstracts into list.
            articles = articles + get_articles(soup)

    # Obtain articles' probabilities according to keywords.
    K_prob = get_Kprob(articles, in_k, ou_k)
    # Sort articles.
    articles, K_prob = sort_rev(articles, K_prob)

    for i, art in enumerate(articles):
        # Title
        title = str(art[1])
        print('\n{}) (P={:.2f}) {}'.format(
            str(len(articles) - i), K_prob[i], textwrap.fill(title, 70)))
        # Authors + arXiv link
        authors = art[0] if len(art[0].split(',')) < 4 else\
            ','.join(art[0].split(',')[:3]) + ', et al.'
        print(textwrap.fill(authors, 77), '\n* ' + str(art[3]) + '\n')
        # Abstract
        print(textwrap.fill(str(art[2]), 80))

    print("\nFinished.")


if __name__ == "__main__":
    main()
