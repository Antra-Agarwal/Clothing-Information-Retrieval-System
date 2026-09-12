import sys
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from src.indexer import build_and_save
from src.retrieval import load_indexes, rank_documents
from src.positional_search import (
    exact_phrase_search,
    ordered_proximity_search
)


DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "tests" / "test_results.txt"


def ensure_indexes_exist():
    required_files = [
        DATA_DIR / "inverted_index.json",
        DATA_DIR / "positional_index.json",
        DATA_DIR / "document_metadata.json"
    ]

    if not all(file.exists() for file in required_files):
        build_and_save()


def format_vsm_results(results):
    if not results:
        return "No matching documents."

    lines = []

    for rank, result in enumerate(results, start=1):
        lines.append(
            f"{rank}. "
            f"{result['doc_id']} | "
            f"{result['title']} | "
            f"{result['category']} | "
            f"score={result['score']:.4f}"
        )

    return "\n".join(lines)


def format_positional_results(results):
    if not results:
        return "No matching documents."

    lines = []

    for rank, result in enumerate(results, start=1):
        lines.append(
            f"{rank}. "
            f"{result['doc_id']} | "
            f"{result['title']} | "
            f"{result['category']} | "
            f"matches={result['matches']}"
        )

    return "\n".join(lines)


def main():

    ensure_indexes_exist()

    inverted_index, positional_index, metadata = load_indexes()

    output = []

    output.append("=" * 80)
    output.append("CLOTHING INFORMATION RETRIEVAL - TEST RESULTS")
    output.append("=" * 80)
    output.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    output.append("")

    # --------------------------------------------------
    # 1. Ten free-text VSM queries
    # --------------------------------------------------

    free_text_queries = [
        "cotton shirt",
        "winter wear",
        "festive clothing",
        "denim jacket",
        "comfortable leggings",
        "regular fit sweatshirt",
        "breathable fabric",
        "hoodie black",
        "high waist",
        "casual travel wear",
        "nonexistentxyzterm"
    ]

    output.append("=" * 80)
    output.append("PART B: FREE-TEXT VSM TESTS")
    output.append("=" * 80)

    for number, query in enumerate(
        free_text_queries,
        start=1
    ):

        results = rank_documents(
            query,
            inverted_index,
            metadata
        )

        output.append("")
        output.append(f"Query {number}: {query}")
        output.append("-" * 80)
        output.append(format_vsm_results(results))

    # --------------------------------------------------
    # 2. Five exact phrase queries
    # --------------------------------------------------

    phrase_queries = [
        "cotton shirt",
        "stretch denim",
        "festive wear",
        "winter wear",
        "regular fit"
    ]

    output.append("")
    output.append("=" * 80)
    output.append("PART C: EXACT PHRASE SEARCH TESTS")
    output.append("=" * 80)

    for number, phrase in enumerate(
        phrase_queries,
        start=1
    ):

        results = exact_phrase_search(
            phrase,
            positional_index,
            metadata
        )

        output.append("")
        output.append(f"Phrase Query {number}: {phrase}")
        output.append("-" * 80)
        output.append(format_positional_results(results))

    # --------------------------------------------------
    # 3. Proximity queries with different k values
    # --------------------------------------------------

    proximity_queries = [
        ("cotton", "shirt", 2),
        ("cotton", "shirt", 3),
        ("stretch", "denim", 4),
        ("winter", "wear", 3)
    ]

    output.append("")
    output.append("=" * 80)
    output.append("PART C: ORDERED PROXIMITY SEARCH TESTS")
    output.append("=" * 80)

    for number, (first, second, k) in enumerate(
        proximity_queries,
        start=1
    ):

        results = ordered_proximity_search(
            first,
            second,
            k,
            positional_index,
            metadata
        )

        output.append("")
        output.append(
            f"Proximity Query {number}: "
            f"{first} WITHIN/{k} {second}"
        )
        output.append("-" * 80)
        output.append(format_positional_results(results))

    # --------------------------------------------------
    # 4. Comparison examples
    # --------------------------------------------------

    output.append("")
    output.append("=" * 80)
    output.append("POSITIONAL VS ORDINARY VSM COMPARISON")
    output.append("=" * 80)

    comparison_queries = [
        "cotton shirt",
        "winter wear"
    ]

    for query in comparison_queries:

        vsm_results = rank_documents(
            query,
            inverted_index,
            metadata
        )

        phrase_results = exact_phrase_search(
            query,
            positional_index,
            metadata
        )

        vsm_docs = [
            result["doc_id"]
            for result in vsm_results
        ]

        phrase_docs = [
            result["doc_id"]
            for result in phrase_results
        ]

        output.append("")
        output.append(f"Comparison Query: {query}")
        output.append(f"VSM Result IDs: {vsm_docs}")
        output.append(f"Exact Phrase IDs: {phrase_docs}")

        if vsm_docs != phrase_docs:
            output.append(
                "Observation: Positional information changes "
                "the result set/order because VSM only considers "
                "term occurrence and weights, whereas phrase "
                "retrieval requires consecutive term positions."
            )
        else:
            output.append(
                "Observation: Both methods returned the same IDs "
                "for this query, but positional retrieval still "
                "verified consecutive term positions."
            )

    OUTPUT_FILE.write_text(
        "\n".join(output),
        encoding="utf-8"
    )

    print("\n".join(output))
    print("\nTest results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()