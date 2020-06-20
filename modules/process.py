
import re
from bs4 import BeautifulSoup as BS
import requests


def arxivDwnld(categ, day_week, mode):
    """
    Downloads data from arXiv.
    """
    if mode == 'recent':
        print("Downloading latest arXiv data.")
        url = "http://arxiv.org/list/" + categ + "/new"
    else:
        year, month, day = day_week
        print("Downloading arXiv data for {}-{}-{} ({})".format(
            year, month, day, categ))
        url = "https://arxiv.org/catchup?smonth=" + month + "&group=grp_&s" +\
              "day=" + day + "&num=50&archive=" + categ +\
              "&method=with&syear=" + year

    html = requests.get(url)
    soup = BS(html.content, 'lxml')

    # with open("temp", "wb") as f:
    #     f.write(html.content)
    # with open("temp", "rb") as f:
    #     soup = BS(f, 'lxml')

    return soup


def articlList(soup):
    """
    Splits articles into lists containing title, abstract, authors and link.
    Article info is located between <dt> and </dd> tags.
    """
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

    subjects = [_.text.replace('\n', ' ') for _
                in soup.find_all(class_="list-subjects")]
    subjects = [re.findall('\((.*?)\)', _) for _ in subjects]

    articles = list(zip(*[authors, titles, abstracts, links, subjects]))

    return articles


def filterSubcats(subcategs, soup_arts):
    """
    Only keep articles whose sub-categories are in the 'subcategs' list.
    """
    date_arts = []
    for art in soup_arts:
        if any(_ in subcategs for _ in art[-1]):
            date_arts.append(art)
    return date_arts
