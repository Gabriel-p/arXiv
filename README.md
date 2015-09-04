# arXiv

Attempts to imitate the [myADS Notification Service](http://myads.harvard.edu/myADS_help.html).

Goes through the new submissions in one ore more
categories in [arXiv](arxiv.org) and presents those
articles more relevant to the user, sorted according
to a list of keywords selected by the user.

Keywords and selected categories are provided to the
script through the `keywords.dat` file. Accepted keywords
move an article upwards the list, while rejected keywords
move it downwards.

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

## Requirements

The following packages are needed (all are included
in the standard installation):

* `urllib`
* `re`
* `shlex`
* `textwrap`
