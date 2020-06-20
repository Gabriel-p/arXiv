from modules import io, rank


def main():
    """
    Query newly added articles to selected arXiv categories, rank them,
    print the ranked list, and ask for manual ranking.
    Ranking:
    1: Primary interest.
    2: Secondary interest.
    3: Tertiary interest.
    4: ....
    """

    # Read date range, arXiv categories, and classification mode.
    mode, date_range, categs, subcategs, groups, gr_ids, clmode = io.readINI()

    # Download articles from arXiv.
    articles, dates = io.downArts(mode, date_range, categs, subcategs)

    if articles:
        # Read previous classifications.
        wordsRank = io.readWords()

        if mode != 'zotero':
            # Obtain articles' probabilities based on ML analysis.
            ranks, probs = rank.probs(clmode, wordsRank, articles)

            # Sort and group articles.
            grpd_arts, gr_len = rank.artSort(
                gr_ids, articles, dates, ranks, probs)

            # Manual ranking.
            train = rank.manual(groups, gr_ids, grpd_arts, gr_len)
        else:
            # Zotero ranking.
            train = rank.zotero(groups, gr_ids, articles, dates)

        # Update classifier data.
        io.updtRank(wordsRank, train)
    else:
        print("No articles found for the date(s) selected.")

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
