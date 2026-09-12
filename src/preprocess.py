import re
from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


# Standard English stop-word list.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with",
    "from", "for", "to", "of", "in", "on", "at", "by", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "as", "into", "about", "over", "after",
    "before", "during", "through", "between", "under", "again",
    "further", "then", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "where", "when", "why", "how"
}


def tokenize(text):
    """
    Convert text into lowercase alphabetic tokens.

    Example:
        "Women's Cotton Shirt - Blue"
        becomes:
        ["women", "s", "cotton", "shirt", "blue"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def normalize_tokens(text):
    """
    Apply tokenization, stop-word removal, and stemming.

    Returns a list of normalized terms.
    """
    raw_tokens = tokenize(text)

    normalized = []

    for token in raw_tokens:
        if token in STOP_WORDS:
            continue

        stemmed = stemmer.stem(token)

        if stemmed:
            normalized.append(stemmed)

    return normalized


def normalize_query(query):
    """
    Normalize a query using the same pipeline as documents.
    """
    return normalize_tokens(query)