import json
import math
import re
from pathlib import Path

from src.preprocess import normalize_tokens
from src.query_expansion import expand_query


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

N_DOCUMENTS = 100


def load_inverted_index():
    """
    Load the inverted index from JSON.
    """
    with open(
        DATA_DIR / "inverted_index.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_positional_index():
    """
    Load the positional index from JSON.
    """
    with open(
        DATA_DIR / "positional_index.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_metadata():
    """
    Load document metadata from JSON.
    """
    with open(
        DATA_DIR / "document_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_indexes():
    """
    Load all indexes and metadata.

    Returns:
        inverted_index, positional_index, metadata
    """
    inverted_index = load_inverted_index()
    positional_index = load_positional_index()
    metadata = load_metadata()

    return inverted_index, positional_index, metadata


def document_number(doc_id):
    """
    Extract the numeric part of a document ID.

    Example:
        D001 -> 1
        D100 -> 100
    """
    match = re.search(r"\d+", str(doc_id))

    if match:
        return int(match.group())

    return float("inf")


def lnc_weight(tf):
    """
    Calculate document-side lnc weighting.

    Formula:
        1 + log10(tf)
    """
    if tf <= 0:
        return 0.0

    return 1 + math.log10(tf)


def ltc_weight(tf, df):
    """
    Calculate query-side ltc weighting.

    Formula:
        (1 + log10(tf)) * log10(N / df)
    """
    if tf <= 0 or df <= 0:
        return 0.0

    idf = math.log10(N_DOCUMENTS / df)

    return (1 + math.log10(tf)) * idf


def normalize_vector(vector):
    """
    Normalize a vector using Euclidean normalization.
    """
    magnitude = math.sqrt(
        sum(weight * weight for weight in vector.values())
    )

    if magnitude == 0:
        return {
            term: 0.0
            for term in vector
        }

    return {
        term: weight / magnitude
        for term, weight in vector.items()
    }


def build_document_vectors(inverted_index):
    """
    Build normalized document vectors using lnc weighting.
    """
    document_vectors = {}

    for term, term_data in inverted_index.items():
        for posting in term_data["postings"]:
            doc_id = posting["doc_id"]
            tf = posting["tf"]

            weight = lnc_weight(tf)

            if doc_id not in document_vectors:
                document_vectors[doc_id] = {}

            document_vectors[doc_id][term] = weight

    for doc_id in document_vectors:
        document_vectors[doc_id] = normalize_vector(
            document_vectors[doc_id]
        )

    return document_vectors


def get_document_vectors(inverted_index=None):
    """
    Get normalized document vectors for the corpus.
    """
    if inverted_index is None:
        inverted_index = load_inverted_index()

    return build_document_vectors(inverted_index)


def build_query_vector(query, inverted_index):
    """
    Build a normalized query vector using ltc weighting.
    """
    query_terms = normalize_tokens(query)

    if not query_terms:
        return {}

    query_tf = {}

    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    query_vector = {}

    for term, tf in query_tf.items():
        if term not in inverted_index:
            continue

        df = inverted_index[term]["df"]

        query_vector[term] = ltc_weight(tf, df)

    return normalize_vector(query_vector)


def cosine_similarity(query_vector, document_vector):
    """
    Calculate cosine similarity between normalized vectors.
    """
    score = 0.0

    for term, query_weight in query_vector.items():
        document_weight = document_vector.get(term, 0.0)

        score += query_weight * document_weight

    return score


def rank_documents(
    query,
    inverted_index=None,
    metadata=None,
    top_k=10
):
    """
    Rank documents using the lnc.ltc Vector Space Model.
    """
    if inverted_index is None:
        inverted_index = load_inverted_index()

    if metadata is None:
        metadata = load_metadata()

    query_vector = build_query_vector(
        query,
        inverted_index
    )

    if not query_vector:
        return []

    document_vectors = build_document_vectors(
        inverted_index
    )

    results = []

    for doc_id, document_vector in document_vectors.items():
        score = cosine_similarity(
            query_vector,
            document_vector
        )

        if score > 0:
            results.append({
                "doc_id": doc_id,
                "title": metadata[doc_id]["title"],
                "category": metadata[doc_id]["category"],
                "score": score
            })

    results.sort(
        key=lambda item: (
            -item["score"],
            document_number(item["doc_id"])
        )
    )

    return results[:top_k]


def search(query, top_k=10):
    """
    Perform free-text Vector Space Model retrieval.
    """
    return rank_documents(
        query,
        top_k=top_k
    )


def vector_space_search(query, top_k=10):
    """
    Alias for compatibility with other modules.
    """
    return search(query, top_k)


if __name__ == "__main__":
    query = input("Enter free-text query: ")

    results = search(query)

    for result in results:
        print(result)

def smart_rank_documents(query, inverted_index=None, metadata=None, top_k=10):
    if inverted_index is None: inverted_index = load_inverted_index()
    if metadata is None: metadata = load_metadata()
    original_terms, expanded = expand_query(query)
    if not original_terms: return []
    vectors = build_document_vectors(inverted_index)
    scores, evidence = {}, {}
    for term, multiplier, reason in expanded:
        if term not in inverted_index: continue
        qw = ltc_weight(1, inverted_index[term]["df"]) * multiplier
        for posting in inverted_index[term]["postings"]:
            doc = posting["doc_id"]
            scores[doc] = scores.get(doc, 0) + qw * vectors.get(doc, {}).get(term, 0)
            evidence.setdefault(doc, []).append((term, reason))
    results = []
    for doc, score in scores.items():
        if score <= 0: continue
        ev = evidence[doc]
        original_matches = sorted({t for t,r in ev if r == "original"})
        expanded_matches = sorted({t for t,r in ev if r != "original"})
        explanation = []
        if original_matches: explanation.append("Original terms matched: " + ", ".join(original_matches))
        if expanded_matches: explanation.append("Expanded clothing terms matched: " + ", ".join(expanded_matches))
        explanation.append(f"{len(ev)} weighted term match(es) contributed to the score")
        results.append({"doc_id":doc,"title":metadata[doc]["title"],"category":metadata[doc]["category"],"score":score,"matched_terms":original_matches,"expanded_terms":expanded_matches,"explanation":explanation})
    results.sort(key=lambda x:(-x["score"], document_number(x["doc_id"])))
    return results[:top_k]
