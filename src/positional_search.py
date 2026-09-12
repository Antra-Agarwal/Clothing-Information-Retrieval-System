import json
import re
from pathlib import Path

from src.preprocess import normalize_query


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_positional_index():
    """
    Load positional index from JSON.
    """
    with open(
        DATA_DIR / "positional_index.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_metadata():
    """
    Load document metadata.
    """
    with open(
        DATA_DIR / "document_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def document_number(doc_id):
    """
    Numeric document sorting.
    """
    match = re.search(r"\d+", doc_id)

    if match:
        return int(match.group())

    return float("inf")


def exact_phrase_search(
    phrase,
    positional_index=None,
    metadata=None
):
    """
    Search for an exact phrase.

    Terms must appear in consecutive positions.

    Example:
        cotton shirt

    If cotton occurs at position 4 and shirt at position 5,
    the phrase matches.
    """

    if positional_index is None:
        positional_index = load_positional_index()

    if metadata is None:
        metadata = load_metadata()

    terms = normalize_query(phrase)

    if not terms:
        return []

    if any(term not in positional_index for term in terms):
        return []

    candidate_docs = None

    for term in terms:
        term_docs = {
            posting["doc_id"]
            for posting in positional_index[term]["postings"]
        }

        if candidate_docs is None:
            candidate_docs = term_docs
        else:
            candidate_docs &= term_docs

    if not candidate_docs:
        return []

    posting_maps = {}

    for term in terms:
        posting_maps[term] = {
            posting["doc_id"]: posting["positions"]
            for posting in positional_index[term]["postings"]
        }

    results = []

    for doc_id in candidate_docs:
        first_positions = posting_maps[terms[0]][doc_id]

        matches = []

        for start_position in first_positions:
            current_position = start_position
            valid = True

            for term in terms[1:]:
                current_position += 1

                if current_position not in posting_maps[term][doc_id]:
                    valid = False
                    break

            if valid:
                matches.append({
                    "start": start_position,
                    "end": current_position
                })

        if matches:
            results.append({
                "doc_id": doc_id,
                "title": metadata[doc_id]["title"],
                "category": metadata[doc_id]["category"],
                "matches": matches
            })

    results.sort(
        key=lambda item: document_number(item["doc_id"])
    )

    return results


def ordered_proximity_search(
    first_term,
    second_term,
    k,
    positional_index=None,
    metadata=None
):
    """
    Ordered proximity search.

    The first term must occur before the second term.

    Condition:
        0 < position(second) - position(first) <= k

    Example:
        cotton WITHIN/3 shirt
    """

    if positional_index is None:
        positional_index = load_positional_index()

    if metadata is None:
        metadata = load_metadata()

    first_terms = normalize_query(first_term)
    second_terms = normalize_query(second_term)

    if not first_terms or not second_terms:
        return []

    first = first_terms[-1]
    second = second_terms[0]

    if first not in positional_index:
        return []

    if second not in positional_index:
        return []

    first_postings = {
        posting["doc_id"]: posting["positions"]
        for posting in positional_index[first]["postings"]
    }

    second_postings = {
        posting["doc_id"]: posting["positions"]
        for posting in positional_index[second]["postings"]
    }

    common_docs = set(first_postings) & set(second_postings)

    results = []

    for doc_id in common_docs:
        matches = []

        first_positions = first_postings[doc_id]
        second_positions = second_postings[doc_id]

        for p1 in first_positions:
            for p2 in second_positions:
                distance = p2 - p1

                if 0 < distance <= k:
                    matches.append({
                        "first_position": p1,
                        "second_position": p2,
                        "distance": distance
                    })

        if matches:
            results.append({
                "doc_id": doc_id,
                "title": metadata[doc_id]["title"],
                "category": metadata[doc_id]["category"],
                "matches": matches
            })

    results.sort(
        key=lambda item: document_number(item["doc_id"])
    )

    return results


def parse_proximity_query(query):
    """
    Parse queries such as:

        cotton WITHIN/3 shirt
        stretch WITHIN/4 denim
    """

    pattern = r"(.+?)\s+WITHIN/(\d+)\s+(.+)"

    match = re.match(
        pattern,
        query.strip(),
        re.IGNORECASE
    )

    if not match:
        return None

    first_term = match.group(1).strip()
    k = int(match.group(2))
    second_term = match.group(3).strip()

    return first_term, second_term, k


def proximity_search(
    query,
    positional_index=None,
    metadata=None
):
    """
    User-friendly proximity search wrapper.

    Example:
        cotton WITHIN/3 shirt
    """

    parsed = parse_proximity_query(query)

    if parsed is None:
        return []

    first_term, second_term, k = parsed

    return ordered_proximity_search(
        first_term,
        second_term,
        k,
        positional_index,
        metadata
    )


if __name__ == "__main__":
    positional_index = load_positional_index()
    metadata = load_metadata()

    print("1. Exact Phrase Search")
    print("2. Proximity Search")

    choice = input("Enter choice: ")

    if choice == "1":
        phrase = input("Enter phrase: ")

        results = exact_phrase_search(
            phrase,
            positional_index,
            metadata
        )

        for result in results:
            print(result)

    elif choice == "2":
        query = input(
            "Enter query (example: cotton WITHIN/3 shirt): "
        )

        results = proximity_search(
            query,
            positional_index,
            metadata
        )

        for result in results:
            print(result)