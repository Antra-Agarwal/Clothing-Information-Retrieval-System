import math

from src.retrieval import (
    load_inverted_index,
    load_metadata,
    build_document_vectors,
    build_query_vector,
    cosine_similarity,
    normalize_vector,
    document_number
)


DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.75
DEFAULT_GAMMA = 0.15


def resolve_feedback_selections(relevant_doc_ids, non_relevant_doc_ids):
    """
    Resolve potential conflicts between relevant and non-relevant selections.

    If a document is marked both relevant and non-relevant, non-relevant
    takes precedence. Returns distinct sets: (relevant_set, non_relevant_set).
    """
    non_relevant_set = set(non_relevant_doc_ids or [])
    relevant_set = set(relevant_doc_ids or []) - non_relevant_set

    return relevant_set, non_relevant_set


def rocchio_reformulate(
    query_vector,
    relevant_doc_ids,
    non_relevant_doc_ids,
    document_vectors,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    gamma=DEFAULT_GAMMA,
    clip_negative=True
):
    """
    Reformulate a query vector using the standard Rocchio algorithm.

    Formula:
        q_new = (alpha * q_current)
              + (beta * (1 / |Dr|) * sum(d in Dr))
              - (gamma * (1 / |Dnr|) * sum(d in Dnr))

    Where:
        Dr: Set of selected relevant documents
        Dnr: Set of selected non-relevant documents

    Args:
        query_vector (dict): Current normalized query vector {term: weight}.
        relevant_doc_ids (list/set): IDs of documents marked relevant.
        non_relevant_doc_ids (list/set): IDs of documents marked non-relevant.
        document_vectors (dict): Corpus document vectors {doc_id: {term: weight}}.
        alpha (float): Weight applied to the current query vector.
        beta (float): Weight applied to centroid of relevant documents.
        gamma (float): Weight applied to centroid of non-relevant documents.
        clip_negative (bool): Whether to clamp negative term weights to 0.0.

    Returns:
        tuple: (new_query_vector, explanation_message)
    """
    if not query_vector:
        return {}, "Empty query vector provided."

    rel_set, non_rel_set = resolve_feedback_selections(
        relevant_doc_ids,
        non_relevant_doc_ids
    )

    if not rel_set and not non_rel_set:
        return (
            dict(query_vector),
            "No feedback selections provided; original query vector retained."
        )

    # Collect valid document vectors present in corpus
    dr_vectors = [
        document_vectors[doc_id]
        for doc_id in rel_set
        if doc_id in document_vectors
    ]

    dnr_vectors = [
        document_vectors[doc_id]
        for doc_id in non_rel_set
        if doc_id in document_vectors
    ]

    # Calculate centroids
    sum_dr = {}
    if dr_vectors:
        for doc in dr_vectors:
            for term, weight in doc.items():
                sum_dr[term] = sum_dr.get(term, 0.0) + weight

    sum_dnr = {}
    if dnr_vectors:
        for doc in dnr_vectors:
            for term, weight in doc.items():
                sum_dnr[term] = sum_dnr.get(term, 0.0) + weight

    # Union of all terms
    all_terms = set(query_vector.keys())
    all_terms.update(sum_dr.keys())
    all_terms.update(sum_dnr.keys())

    dr_count = len(dr_vectors)
    dnr_count = len(dnr_vectors)

    new_weights = {}

    for term in all_terms:
        q_term = query_vector.get(term, 0.0)
        dr_term = (sum_dr.get(term, 0.0) / dr_count) if dr_count > 0 else 0.0
        dnr_term = (sum_dnr.get(term, 0.0) / dnr_count) if dnr_count > 0 else 0.0

        weight = (alpha * q_term) + (beta * dr_term) - (gamma * dnr_term)

        if clip_negative:
            weight = max(0.0, weight)

        if weight > 0:
            new_weights[term] = weight

    # Check magnitude before or after normalization
    magnitude = math.sqrt(
        sum(w * w for w in new_weights.values())
    )

    if magnitude == 0.0:
        return (
            dict(query_vector),
            "Reformulated query vector has zero magnitude; previous query vector retained."
        )

    normalized_new_vector = normalize_vector(new_weights)

    message = (
        f"Query refined using {len(rel_set)} relevant and "
        f"{len(non_rel_set)} non-relevant selection(s)."
    )

    return normalized_new_vector, message


def rank_by_vector(
    query_vector,
    document_vectors=None,
    metadata=None,
    inverted_index=None,
    top_k=10
):
    """
    Rank all documents in the corpus against a query vector using cosine similarity.

    Args:
        query_vector (dict): Normalized query vector {term: weight}.
        document_vectors (dict, optional): Corpus document vectors.
        metadata (dict, optional): Document metadata.
        inverted_index (dict, optional): Inverted index.
        top_k (int): Number of top results to return.

    Returns:
        list: Top-k document dictionaries with doc_id, title, category, score.
    """
    if not query_vector:
        return []

    if inverted_index is None:
        inverted_index = load_inverted_index()

    if document_vectors is None:
        document_vectors = build_document_vectors(inverted_index)

    if metadata is None:
        metadata = load_metadata()

    results = []

    for doc_id, document_vector in document_vectors.items():
        score = cosine_similarity(query_vector, document_vector)

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


def apply_relevance_feedback(
    current_query_vector,
    relevant_doc_ids,
    non_relevant_doc_ids,
    inverted_index=None,
    metadata=None,
    document_vectors=None,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    gamma=DEFAULT_GAMMA,
    original_results=None,
    top_k=10
):
    """
    Apply a round of Rocchio relevance feedback and re-rank the corpus.

    Supports sequential feedback iterations where current_query_vector can be
    either the initial normalized query vector or the result of a previous round.

    Args:
        current_query_vector (dict): The query vector from the current round.
        relevant_doc_ids (list/set): Selected relevant document IDs.
        non_relevant_doc_ids (list/set): Selected non-relevant document IDs.
        inverted_index (dict, optional): Inverted index.
        metadata (dict, optional): Document metadata.
        document_vectors (dict, optional): Document vectors.
        alpha (float): Rocchio alpha parameter (default: 1.0).
        beta (float): Rocchio beta parameter (default: 0.75).
        gamma (float): Rocchio gamma parameter (default: 0.15).
        original_results (list, optional): Pre-computed results to fall back on.
        top_k (int): Number of results to return (default: 10).

    Returns:
        tuple: (results, new_query_vector, message)
    """
    if inverted_index is None:
        inverted_index = load_inverted_index()

    if document_vectors is None:
        document_vectors = build_document_vectors(inverted_index)

    if metadata is None:
        metadata = load_metadata()

    rel_set, non_rel_set = resolve_feedback_selections(
        relevant_doc_ids,
        non_relevant_doc_ids
    )

    if not rel_set and not non_rel_set:
        fallback_results = (
            original_results
            if original_results is not None
            else rank_by_vector(
                current_query_vector,
                document_vectors,
                metadata,
                inverted_index,
                top_k
            )
        )
        return (
            fallback_results,
            current_query_vector,
            "No feedback selections provided; ranking unchanged."
        )

    new_vector, message = rocchio_reformulate(
        current_query_vector,
        rel_set,
        non_rel_set,
        document_vectors,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        clip_negative=True
    )

    # If new_vector is identical or empty due to zero magnitude
    if not new_vector or new_vector == current_query_vector:
        fallback_results = (
            original_results
            if original_results is not None
            else rank_by_vector(
                current_query_vector,
                document_vectors,
                metadata,
                inverted_index,
                top_k
            )
        )
        return fallback_results, current_query_vector, message

    new_results = rank_by_vector(
        new_vector,
        document_vectors=document_vectors,
        metadata=metadata,
        inverted_index=inverted_index,
        top_k=top_k
    )

    return new_results, new_vector, message
