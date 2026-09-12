# Clothing Information Retrieval System

A complete Information Retrieval (IR) system for searching a collection
of 100 clothing-product descriptions using:

-   Text preprocessing and stemming
-   An inverted index
-   TF-IDF cosine-similarity retrieval
-   A positional index
-   Exact phrase search
-   Ordered proximity search
-   Command-line and Streamlit interfaces
-   Automated evaluation and comparison tests
-   Explainable clothing-domain query expansion (Novelty).

## GitHub Repository

[View the complete source code on GitHub](https://github.com/Antra-Agarwal/Clothing-Information-Retrieval-System.git)

------------------------------------------------------------------------

## 1. Project Overview

This project implements a clothing search engine over 100 product
documents. Users can enter free-text queries and retrieve relevant
products ranked by cosine similarity. The system also supports
positional retrieval, allowing users to search for exact phrases and
terms occurring within a specified distance.

The implementation follows the assignment requirements for Parts A--E
and extends the system with an explainable smart clothing retrieval
feature.

  -----------------------------------------------------------------------
  Part                    Requirement             Implementation
  ----------------------- ----------------------- -----------------------
  A                       Preprocessing and       Tokenization, stop-word
                          indexing                removal, stemming,
                                                  inverted index,
                                                  positional index

  B                       Vector-space retrieval  `lnc.ltc` weighting,
                                                  normalization, cosine
                                                  similarity, top-10
                                                  ranking

  C                       Positional retrieval    Exact phrase and
                                                  ordered proximity
                                                  search

  D                       User interface          CLI and Streamlit web
                                                  application

  E                       Testing and analysis    Free-text, phrase,
                                                  proximity,
                                                  unknown-term, and
                                                  before/after comparison
                                                  tests

  Novelty                 Smart clothing          Clothing synonym
                          retrieval               expansion, weighted
                                                  relevance ranking, and
                                                  result explanations
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 2. Features

### Free-Text VSM Retrieval

-   Accepts natural-language queries such as `cotton shirt`.
-   Uses the `lnc.ltc` weighting scheme.
-   Ranks documents using cosine similarity.
-   Returns up to 10 results.
-   Displays document ID, title, category, and score.
-   Uses numeric document ID as the tie-breaker.

### Exact Phrase Search

-   Accepts queries such as `cotton shirt`.
-   Requires all query terms to occur consecutively and in the same
    order.
-   Returns matching document positions.
-   Searches the complete indexed document, including the title and
    description.

### Ordered Proximity Search

-   Accepts queries such as:

    ``` text
    cotton WITHIN/3 shirt
    ```

-   Requires the first term to occur before the second term.

-   Allows the distance between terms to be controlled by `k`.

-   Displays the matching positions and distances.

### Automated Testing

The test suite includes:

-   11 free-text queries
-   5 exact phrase queries
-   4 ordered proximity queries with different window sizes
-   An unknown-term query
-   VSM versus exact-phrase comparisons
-   Position-based proximity comparisons
-   Output saved to `tests/test_results.txt`

### Explainable Smart Clothing Retrieval --- Novelty

The novelty feature extends ordinary retrieval using a small
clothing-specific synonym dictionary.

  Original Term   Related Terms
  --------------- -----------------------
  shirt           top, blouse, tee
  pant            trouser, jean, bottom
  shoe            footwear, sneaker
  dress           gown, frock
  jacket          coat, blazer
  sweatshirt      hoodie, pullover

The smart retrieval mode:

-   Preserves original query terms.
-   Expands selected clothing terms with related vocabulary.
-   Gives original terms full weight.
-   Gives synonym terms a lower weight.
-   Ranks documents using weighted relevance contributions.
-   Explains why each result matched.
-   Displays original and expanded matching terms.

The original Boolean, phrase, and proximity retrieval functionality
remains unchanged.

------------------------------------------------------------------------

## 3. Project Structure

``` text
Clothing_IR_Assignment/

│
├── app/
│   └── app.py
│
├── data/
│   ├── document_metadata.json
│   ├── inverted_index.json
│   └── positional_index.json
│
├── screenshots/
│   ├── 01_streamlit_home.png
│   ├── 02_free_text_search.png
│   ├── 03_exact_phrase_search.png
│   ├── 04_proximity_search.png
│   ├── 05_cli_free_text.png
│   ├── 06_cli_exact_phrase.png
│   ├── 07_cli_proximity_within_3.png
│   ├── 08_cli_proximity_within_1.png
│   ├── 09_cli_proximity_within_5.png
│   └── 10_smart_search_novelty.png
│
├── src/
│   ├── __init__.py
│   ├── indexer.py
│   ├── main.py
│   ├── positional_search.py
│   ├── preprocess.py
│   ├── query_expansion.py
│   └── retrieval.py
│
├── tests/
│   ├── run_tests.py
│   └── test_results.txt
│
├── corpus_100.txt
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 4. Dataset

The corpus contains 100 clothing-product documents.

Each document contains:

-   `DOCID`
-   Product title
-   Product category
-   Product description

The indexer reads the corpus and stores document metadata separately in:

``` text
data/document_metadata.json
```

The title and description are combined before preprocessing.
Consequently, phrase and proximity searches can match terms appearing in
either the title or the description.

------------------------------------------------------------------------

## 5. Text Preprocessing

All documents and queries pass through the same preprocessing pipeline.

### Processing Steps

1.  Convert text to lowercase.
2.  Remove punctuation and non-word separators.
3.  Tokenize text into individual terms.
4.  Remove a manually defined English stop-word list.
5.  Apply the Porter stemming algorithm.

Example:

``` text
Original:
Women's Cotton Shirts!

After preprocessing:
womens cotton shirt
```

The same preprocessing is applied consistently to both documents and
user queries so that indexed terms and query terms use the same
representation.

### Stop-word Policy

A manually defined English stop-word list is used. The policy is kept
consistent for both indexing and retrieval.

### Stemming

NLTK's `PorterStemmer` is used to reduce related word forms to a common
stem.

------------------------------------------------------------------------

## 6. Inverted Index

The inverted index maps each normalized term to the documents in which
it occurs.

Conceptually:

``` text
term → document frequency → postings list
```

Example structure:

``` json
{
  "cotton": {
    "df": 20,
    "postings": [
      ["D001", 2],
      ["D002", 2]
    ]
  }
}
```

The actual generated index is stored in:

``` text
data/inverted_index.json
```

The postings list stores document IDs and term frequencies.

------------------------------------------------------------------------

## 7. Positional Index

The positional index stores the positions at which each term occurs in
each document.

Conceptually:

``` text
term → document frequency → postings
```

Each posting contains:

``` text
(document ID, term frequency, positions)
```

Example:

``` json
{
  "cotton": {
    "df": 20,
    "postings": [
      {
        "doc_id": "D001",
        "tf": 2,
        "positions": [3, 18]
      }
    ]
  }
}
```

The actual generated positional index is stored in:

``` text
data/positional_index.json
```

Positions are generated after preprocessing and include terms from both
the title and description.

------------------------------------------------------------------------

## 8. Vector-Space Retrieval Model

The system uses the `lnc.ltc` weighting scheme.

Let:

-   `N = 100` be the total number of documents.
-   `tf(t,d)` be the frequency of term `t` in document `d`.
-   `df(t)` be the number of documents containing term `t`.

### Document Weight: `lnc`

For a term occurring in a document:

``` text
w(t,d) = 1 + log10(tf(t,d))
```

No IDF component is used for document weights.

After assigning weights, the document vector is normalized using its
Euclidean length:

``` text
||d|| = sqrt(sum(w(t,d)^2))
```

### Query Weight: `ltc`

The query uses logarithmic term frequency and inverse document
frequency:

``` text
w(t,q) =
    (1 + log10(tf(t,q))) × log10(N / df(t))
```

The query vector is also cosine-normalized.

### Cosine Similarity

The similarity between a query vector and a document vector is:

``` text
cosine(q,d) = (q · d) / (||q|| × ||d||)
```

Documents are ranked in descending order of cosine similarity. When
scores are equal, the numeric document ID is used as the tie-breaker.

Only the top 10 results are returned by ordinary VSM retrieval.

------------------------------------------------------------------------

## 9. Exact Phrase Retrieval

Exact phrase retrieval uses the positional index instead of only term
frequencies.

For a query such as:

``` text
cotton shirt
```

a document matches only when:

1.  Both normalized terms occur in the document.
2.  `shirt` occurs immediately after `cotton`.
3.  The terms occur in the same order.

If a match exists, the system reports the starting and ending positions.

Example:

``` text
matches=[
  {"start": 3, "end": 4},
  {"start": 9, "end": 10}
]
```

A term appearing alone is not sufficient for an exact phrase match.

------------------------------------------------------------------------

## 10. Ordered Proximity Retrieval

Ordered proximity retrieval extends phrase matching by allowing a
configurable distance.

Example:

``` text
cotton WITHIN/3 shirt
```

A document matches when:

``` text
0 < position(shirt) - position(cotton) <= 3
```

This means:

-   The first term must occur before the second term.
-   The maximum allowed distance is 3 positions.
-   The actual matching positions and distances are displayed.

Example:

``` text
matches=[
  {
    "first_position": 15,
    "second_position": 18,
    "distance": 3
  }
]
```

Increasing the proximity window makes the search less restrictive and
may add documents that were not returned for a smaller window.

------------------------------------------------------------------------

## 11. Novelty Feature: Explainable Smart Clothing Retrieval

The project adds a clothing-domain query expansion layer on top of the
ordinary retrieval system.

### How It Works

When a user enters a clothing-related query, the system:

1.  Normalizes the original query.
2.  Identifies clothing terms with known synonyms.
3.  Adds related terms using a predefined synonym dictionary.
4.  Assigns full weight to original terms.
5.  Assigns lower weight to synonym terms.
6.  Scores documents using the weighted term matches.
7.  Displays an explanation for each result.

For example:

``` text
Query:
black shirt
```

The system searches for the original terms while also considering
related clothing vocabulary.

### Example Smart Search Terms

``` text
shirt → top, blouse, tee
shoe → footwear, sneaker
jacket → coat, blazer
sweatshirt → hoodie, pullover
```

### Result Explanation

Each smart-search result can display:

-   Smart relevance score
-   Original query terms that matched
-   Expanded synonym terms that matched
-   Number of weighted term matches contributing to the score

This makes the retrieval process more transparent and demonstrates
explainable search.

------------------------------------------------------------------------

## 12. User Interfaces

### Command-Line Interface

Run:

``` bash
python -m src.main
```

The CLI provides options for:

1.  Free-text VSM search
2.  Exact phrase search
3.  Ordered proximity search
4.  Exit

The program automatically builds the indexes if they do not already
exist.

### Streamlit Web Interface

Run:

``` bash
streamlit run app/app.py
```

If Streamlit is not recognized, use:

``` bash
python -m streamlit run app/app.py
```

The web application provides:

-   A search-mode selector
-   A query input box
-   Free-text VSM retrieval
-   Exact phrase retrieval
-   Ordered proximity retrieval
-   Smart Clothing Search (Novelty)
-   Result scores
-   Document metadata
-   Matching positions and distances for positional searches
-   Original and expanded matching terms for smart retrieval
-   Corpus and index statistics in the sidebar

------------------------------------------------------------------------

## 13. Installation

### Prerequisites

-   Python 3.9 or later
-   `pip`

### Create a Virtual Environment

``` bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows PowerShell

``` powershell
.\venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

``` cmd
venv\Scripts\activate
```

#### Linux/macOS

``` bash
source venv/bin/activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

The requirements are:

``` text
streamlit
nltk
```

`PorterStemmer` does not require downloading additional NLTK datasets.

------------------------------------------------------------------------

## 14. Building the Indexes

To build or rebuild the indexes from the corpus:

``` bash
python -m src.indexer
```

The generated files are:

``` text
data/document_metadata.json
data/inverted_index.json
data/positional_index.json
```

The system also builds the indexes automatically when required by the
CLI or Streamlit application.

------------------------------------------------------------------------

## 15. Running the Tests

Run:

``` bash
python tests/run_tests.py
```

The test output is saved to:

``` text
tests/test_results.txt
```

The report contains separate sections for:

-   Part B: Free-text VSM tests
-   Part C: Exact phrase search tests
-   Part C: Ordered proximity search tests
-   Positional versus ordinary VSM comparison

------------------------------------------------------------------------

## 16. Test Results Summary

### Free-text VSM

The test suite successfully demonstrates:

-   Relevant product retrieval for normal queries.
-   Top-10 ranking with cosine scores.
-   Correct handling of queries with no matching documents.
-   Unknown-term handling.

Example:

``` text
Query: cotton shirt

1. D022 | Men's Checked Cotton Shirt - White | Shirt | score=0.2624
2. D082 | Men's Checked Cotton Shirt - Blue | Shirt | score=0.2624
3. D001 | Men's Cotton Crew Neck T-Shirt - Black | T-Shirt | score=0.2613
```

### Exact Phrase Search

For:

``` text
cotton shirt
```

ordinary VSM returns documents containing the terms, while exact phrase
retrieval keeps only documents where the terms occur consecutively.

Example result IDs:

``` text
VSM:

D022, D082, D001, D041, D061, D042, D081, D002, D062, D021

Exact phrase:

D002, D022, D042, D062, D082
```

This demonstrates the effect of positional information.

### Proximity Search

For:

``` text
cotton WITHIN/2 shirt
```

the system returns documents where the terms are within two positions.

When the query changes to:

``` text
cotton WITHIN/3 shirt
```

additional documents can appear because the allowed distance is larger.

For example, a document with:

``` text
cotton at position 15
shirt at position 18
distance = 3
```

is excluded from `WITHIN/2` but included in `WITHIN/3`.

------------------------------------------------------------------------

## 17. VSM versus Positional Retrieval

Ordinary VSM and positional retrieval solve different search
requirements.

  Feature                          Ordinary VSM       Exact Phrase   Ordered Proximity
  -------------------------------- ------------------ -------------- -------------------
  Uses term frequency              Yes                No             No
  Uses IDF                         Yes, for queries   No             No
  Uses term positions              No                 Yes            Yes
  Requires consecutive terms       No                 Yes            No
  Supports configurable distance   No                 No             Yes
  Returns cosine scores            Yes                No             No
  Returns matching positions       No                 Yes            Yes

### Example: `cotton shirt`

VSM ranks documents based on term occurrence and weighting. It does not
require the words to appear together.

Exact phrase retrieval requires:

``` text
cotton shirt
```

to occur consecutively.

Therefore, positional retrieval can change both the result set and the
ordering of results.

### Example: Proximity Window

Changing:

``` text
cotton WITHIN/2 shirt
```

to:

``` text
cotton WITHIN/3 shirt
```

allows additional documents where the two terms are slightly farther
apart.

------------------------------------------------------------------------

## 18. Screenshots

The `screenshots/` directory contains visual evidence of the implemented
features.

### 1. Streamlit Home Interface

![Streamlit Home Interface](screenshots/01_streamlit_home.png)

This screenshot demonstrates the main Streamlit interface, search-mode
selector, and application layout.

### 2. Free-Text Search

![Free-Text Search](screenshots/02_free_text_search.png)

This demonstrates ordinary free-text retrieval using the vector-space
model.

### 3. Exact Phrase Search

![Exact Phrase Search](screenshots/03_exact_phrase_search.png)

This demonstrates exact phrase retrieval using positional information.

### 4. Proximity Search

![Proximity Search](screenshots/04_proximity_search.png)

This demonstrates ordered proximity retrieval with a configurable
distance.

### 5. CLI Free-Text Search

![CLI Free-Text Search](screenshots/05_cli_free_text.png)

This screenshot demonstrates free-text retrieval through the
command-line interface.

### 6. CLI Exact Phrase Search

![CLI Exact Phrase Search](screenshots/06_cli_exact_phrase.png)

This demonstrates exact phrase retrieval through the CLI.

### 7. CLI Proximity Search Within 3

![CLI Proximity Within 3](screenshots/07_cli_proximity_within_3.png)

This demonstrates ordered proximity retrieval using a window of three
positions.

### 8. Proximity Search Within 1

![Proximity Within 1](screenshots/08_proximity_within_1.png)

This demonstrates a stricter proximity window.

### 9. Proximity Search Within 5

![Proximity Within 5](screenshots/09_proximity_within_5.png)

This demonstrates a larger proximity window and the effect of relaxed
distance constraints.

### 10. Smart Clothing Search --- Novelty Feature

![Smart Clothing Search
Novelty](screenshots/10_smart_search_novelty.png)

This screenshot demonstrates the novelty feature, including clothing
synonym expansion, weighted smart relevance scoring, and human-readable
explanations for retrieved results.

------------------------------------------------------------------------

## 19. Source Files

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `src/preprocess.py`                 Lowercasing, punctuation removal,
                                      tokenization, stop-word removal,
                                      and stemming

  `src/indexer.py`                    Corpus parsing and
                                      inverted/positional index
                                      construction

  `src/retrieval.py`                  `lnc.ltc` weighting and
                                      cosine-similarity retrieval

  `src/query_expansion.py`            Clothing synonym dictionary and
                                      query expansion

  `src/positional_search.py`          Exact phrase and ordered proximity
                                      retrieval

  `src/main.py`                       Command-line interface

  `app/app.py`                        Streamlit graphical interface

  `tests/run_tests.py`                Automated evaluation and comparison
                                      tests
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 20. Limitations

-   Retrieval depends on the terms present in the corpus.
-   Unknown or absent terms produce no matching documents.
-   Stemming may make some words share a normalized representation.
-   Exact phrase and proximity searches operate on the preprocessed
    token positions.
-   Ordinary VSM retrieval is limited to the top 10 results, while
    positional test reports may display more matches for analysis.
-   Smart retrieval depends on the predefined clothing synonym
    dictionary.
-   Query expansion is conservative and does not use an external
    language model or live knowledge source.

------------------------------------------------------------------------

## 21. Conclusion

The Clothing Information Retrieval System provides a complete search
workflow over a 100-document clothing corpus. It combines conventional
vector-space retrieval with positional retrieval to support both
relevance-based searching and structure-aware queries.

The project demonstrates:

-   Consistent document/query preprocessing
-   Inverted and positional index construction
-   `lnc.ltc` cosine-based retrieval
-   Exact phrase matching
-   Ordered proximity matching
-   Interactive CLI and Streamlit search interfaces
-   Automated testing and before/after retrieval analysis
-   Explainable clothing-domain query expansion

The novelty feature improves search flexibility by considering related
clothing vocabulary while preserving the original query terms and
explaining why each document was retrieved.

The implementation is organized so that the generated indexes, source
code, tests, screenshots, and documentation can be submitted together as
the final assignment deliverable.
