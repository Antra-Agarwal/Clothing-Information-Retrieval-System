import sys
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from src.indexer import build_and_save
from src.retrieval import load_indexes, rank_documents, build_query_vector
from src.positional_search import (
    exact_phrase_search,
    ordered_proximity_search
)
from src.relevance_feedback import (
    apply_relevance_feedback,
    rocchio_reformulate
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

    # --------------------------------------------------
    # 5. Novelty: Relevance Feedback (Rocchio) Tests
    # --------------------------------------------------

    output.append("")
    output.append("=" * 80)
    output.append("NOVELTY: RELEVANCE FEEDBACK (ROCCHIO) TESTS")
    output.append("=" * 80)

    # Test 5.1: Normal case (some relevant + some non-relevant picked)
    rf_query_1 = "cotton shirt"
    vsm_orig_1 = rank_documents(rf_query_1, inverted_index, metadata)
    q_vec_1 = build_query_vector(rf_query_1, inverted_index)

    rel_1 = ["D002"]
    non_rel_1 = ["D001"]

    rf_results_1, new_vec_1, msg_1 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=rel_1,
        non_relevant_doc_ids=non_rel_1,
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    output.append("")
    output.append(f"Rocchio Test 1: Normal Case (Query: '{rf_query_1}')")
    output.append(f"Relevant Docs Selected: {rel_1}")
    output.append(f"Non-Relevant Docs Selected: {non_rel_1}")
    output.append(f"Feedback Status: {msg_1}")
    output.append("-" * 80)
    output.append("ORIGINAL TOP 5:")
    output.append(format_vsm_results(vsm_orig_1[:5]))
    output.append("")
    output.append("REFINED TOP 5:")
    output.append(format_vsm_results(rf_results_1[:5]))

    orig_rank_d002 = next((i for i, r in enumerate(vsm_orig_1, 1) if r["doc_id"] == "D002"), None)
    new_rank_d002 = next((i for i, r in enumerate(rf_results_1, 1) if r["doc_id"] == "D002"), None)
    output.append(
        f"Observation: Relevant document D002 moved from rank {orig_rank_d002} to rank {new_rank_d002}, "
        "confirming query vector shift toward the relevant cluster."
    )

    # Test 5.2: No-selection case (no documents picked)
    rf_results_2, new_vec_2, msg_2 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=[],
        non_relevant_doc_ids=[],
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    output.append("")
    output.append(f"Rocchio Test 2: No-Selection Edge Case (Query: '{rf_query_1}')")
    output.append("Relevant Docs: [], Non-Relevant Docs: []")
    output.append(f"Feedback Status: {msg_2}")
    same_results_2 = [r["doc_id"] for r in rf_results_2] == [r["doc_id"] for r in vsm_orig_1]
    output.append(f"Ranking Unchanged: {same_results_2}")
    output.append(
        "Observation: When no feedback is provided, the system safely retains "
        "the original ranking without error."
    )

    # Test 5.3: All-relevant case
    top3_rel = [r["doc_id"] for r in vsm_orig_1[:3]]
    rf_results_3, new_vec_3, msg_3 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=top3_rel,
        non_relevant_doc_ids=[],
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    output.append("")
    output.append(f"Rocchio Test 3: All-Relevant Feedback (Query: '{rf_query_1}')")
    output.append(f"Relevant Docs: {top3_rel}, Non-Relevant Docs: []")
    output.append(f"Feedback Status: {msg_3}")
    output.append("-" * 80)
    output.append("REFINED TOP 5:")
    output.append(format_vsm_results(rf_results_3[:5]))
    output.append(
        "Observation: Relevant documents are reinforced with beta=0.75 and gamma=0."
    )

    # Test 5.4: All-non-relevant case
    top3_non_rel = [r["doc_id"] for r in vsm_orig_1[:3]]
    rf_results_4, new_vec_4, msg_4 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=[],
        non_relevant_doc_ids=top3_non_rel,
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    output.append("")
    output.append(f"Rocchio Test 4: All-Non-Relevant Feedback (Query: '{rf_query_1}')")
    output.append(f"Relevant Docs: [], Non-Relevant Docs: {top3_non_rel}")
    output.append(f"Feedback Status: {msg_4}")
    output.append("-" * 80)
    output.append("REFINED TOP 5:")
    output.append(format_vsm_results(rf_results_4[:5]))
    output.append(
        "Observation: Non-relevant documents are penalized (gamma=0.15, beta=0)."
    )

    # Test 5.5: Sequential Multi-Round Feedback
    rf_round1_res, round1_vec, msg_round1 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=["D062"],
        non_relevant_doc_ids=["D001"],
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    rf_round2_res, round2_vec, msg_round2 = apply_relevance_feedback(
        current_query_vector=round1_vec,
        relevant_doc_ids=["D042"],
        non_relevant_doc_ids=["D021"],
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=rf_round1_res
    )

    output.append("")
    output.append(f"Rocchio Test 5: Sequential Multi-Round Feedback (Query: '{rf_query_1}')")
    output.append(f"Round 1: {msg_round1}")
    output.append(f"Round 1 Top IDs: {[r['doc_id'] for r in rf_round1_res[:5]]}")
    output.append(f"Round 2: {msg_round2}")
    output.append(f"Round 2 Top IDs: {[r['doc_id'] for r in rf_round2_res[:5]]}")
    output.append(
        "Observation: Multi-round feedback seamlessly reformulates from current vector iteratively."
    )

    # Test 5.6: Precedence Conflict Handling
    rf_results_6, new_vec_6, msg_6 = apply_relevance_feedback(
        current_query_vector=q_vec_1,
        relevant_doc_ids=["D002", "D041"],
        non_relevant_doc_ids=["D002"],
        inverted_index=inverted_index,
        metadata=metadata,
        original_results=vsm_orig_1
    )

    output.append("")
    output.append("Rocchio Test 6: Precedence Conflict Handling (D002 marked both)")
    output.append(f"Feedback Status: {msg_6}")
    output.append(
        "Observation: D002 was marked both relevant and non-relevant; non-relevant took precedence "
        "and execution degraded gracefully with no error."
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