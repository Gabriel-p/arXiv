# arXiv

Article classifier, originally based on the
[myADS Notification Service](http://myads.harvard.edu/myADS_help.html).

Goes through the submissions in one ore more categories in [arXiv](arxiv.org)
and presents those articles more relevant on top.

The sorting is performed according to a list of ranked articles generated
by the user through continued use. The more you use the code, the more it will
learn about your preferred articles.

Ranked data is stored as a plain text csv file called `classifier.dat` with two
columns: the ranking assigned, and article data used to classify it (title and
abstract).

The ranking is done assigning an integer from `1, 2, 3` to an article, where
`1` means *"not interested"*, `2` means *"neutral"* and `3` means
*"interested"*.

The predictions (automatic classification) is performed via either of these
methods:
- `NB`: [Naive Bayes classifier for multinomial models](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Linear classifiers with SGD training](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
  - `SVM`: linear SVM
  - `LR`: logistic regression
  - `MH`: smooth loss that brings tolerance to outliers as well as probability
          estimates (Modified Huber)
  - `PC`: linear loss used by the perceptron algorithm


## Categories

Astrophysics: **astro-ph**

Condensed Matter: **cond-mat**

General Relativity and Quantum Cosmology: **gr-qc**

High Energy Physics - Experiment: **hep-ex**

High Energy Physics - Lattice: **hep-lat**

High Energy Physics - Phenomenology: **hep-ph**

High Energy Physics - Theory: **hep-th**

Mathematical Physics: **math-ph**

Nonlinear Sciences: **nlin**

Nuclear Experiment: **nucl-ex**

Nuclear Theory: **nucl-th**

Physics: **physics**

Quantum Physics: **quant-ph**

Mathematics: **math**

Computing Research Repository: **CoRR**

Quantitative Biology: **q-bio**

Quantitative Finance: **q-fin**

Statistics: **stat**

## Installing

The [conda](https://conda.io/) environment can be created with

`$ conda create -n arxivenv scikit-learn pandas numpy beautifulsoup4 requests lxml`

and activated with:

`$ source activate arxivenv`
