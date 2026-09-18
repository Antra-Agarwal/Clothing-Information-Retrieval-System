import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from src.indexer import build_and_save
from src.retrieval import (
    load_indexes,
    rank_documents,
    smart_rank_documents,
    build_query_vector
)
from src.positional_search import (
    exact_phrase_search,
    ordered_proximity_search,
    parse_proximity_query
)
from src.relevance_feedback import apply_relevance_feedback


DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="Clothing IR Search Engine",
    page_icon="👕",
    layout="wide"
)


@st.cache_data
def prepare_indexes():
    """
    Build indexes if they do not exist.
    """

    required_files = [
        DATA_DIR / "inverted_index.json",
        DATA_DIR / "positional_index.json",
        DATA_DIR / "document_metadata.json"
    ]

    if not all(file.exists() for file in required_files):
        build_and_save()


prepare_indexes()

inverted_index, positional_index, metadata = load_indexes()


st.title("👕 Clothing Information Retrieval Search Engine")

st.markdown(
    """
    This application implements:

    - Inverted Index
    - lnc.ltc Vector Space Model
    - Cosine Similarity Ranking
    - Positional Index
    - Exact Phrase Search
    - Ordered Proximity Search
    """
)


st.sidebar.header("Search Mode")

mode = st.sidebar.selectbox(
    "Select search mode",
    [
        "Free-text VSM Search",
        "Exact Phrase Search",
        "Ordered Proximity Search",
        "Smart Clothing Search (Novelty)"
    ]
)


if mode == "Free-text VSM Search":

    st.header("Free-text Ranked Retrieval")

    query = st.text_input(
        "Enter a clothing query",
        placeholder="Example: cotton shirt"
    )

    if st.button("Search", type="primary"):

        if not query.strip():
            st.warning("Please enter a query.")
            st.session_state.pop("vsm_active_query", None)
            st.session_state.pop("vsm_original_results", None)
            st.session_state.pop("vsm_feedback_rounds", None)
        else:
            keys_to_clear = [
                k for k in st.session_state if k.startswith("vsm_fb_")
            ]
            for k in keys_to_clear:
                del st.session_state[k]

            results = rank_documents(
                query,
                inverted_index,
                metadata
            )
            q_vector = build_query_vector(query, inverted_index)

            st.session_state["vsm_active_query"] = query
            st.session_state["vsm_original_results"] = results
            st.session_state["vsm_current_vector"] = q_vector
            st.session_state["vsm_feedback_rounds"] = []

    if st.session_state.get("vsm_active_query"):
        orig_results = st.session_state.get("vsm_original_results", [])

        if not orig_results:
            st.error(
                "No matching documents found. "
                "Try another query."
            )
        else:
            feedback_rounds = st.session_state.get("vsm_feedback_rounds", [])
            round_count = len(feedback_rounds)

            st.subheader("Original Results")

            for rank, result in enumerate(orig_results, start=1):

                with st.container(border=True):

                    col1, col2, col3 = st.columns([1, 3.5, 2.5])

                    with col1:
                        st.metric(
                            "Rank",
                            rank
                        )

                        st.metric(
                            "Cosine Score",
                            f"{result['score']:.4f}"
                        )

                    with col2:
                        st.markdown(
                            f"### {result['title']}"
                        )

                        st.write(
                            f"**Document ID:** "
                            f"{result['doc_id']}"
                        )

                        st.write(
                            f"**Category:** "
                            f"{result['category']}"
                        )

                    with col3:
                        st.write("**Relevance Feedback:**")
                        st.radio(
                            "Feedback",
                            ["No opinion", "Relevant", "Not relevant"],
                            horizontal=True,
                            key=f"vsm_fb_{result['doc_id']}_r0",
                            label_visibility="collapsed"
                        )

            if round_count == 0:
                if st.button("Refine results with feedback", type="primary", key="btn_refine_initial"):
                    rel_ids = []
                    non_rel_ids = []
                    for r in orig_results:
                        val = st.session_state.get(f"vsm_fb_{r['doc_id']}_r0", "No opinion")
                        if val == "Relevant":
                            rel_ids.append(r["doc_id"])
                        elif val == "Not relevant":
                            non_rel_ids.append(r["doc_id"])

                    if not rel_ids and not non_rel_ids:
                        st.warning("Please mark at least one document as Relevant or Not relevant.")
                    else:
                        new_res, new_vec, msg = apply_relevance_feedback(
                            current_query_vector=st.session_state["vsm_current_vector"],
                            relevant_doc_ids=rel_ids,
                            non_relevant_doc_ids=non_rel_ids,
                            inverted_index=inverted_index,
                            metadata=metadata,
                            original_results=orig_results
                        )
                        st.session_state["vsm_current_vector"] = new_vec
                        st.session_state["vsm_feedback_rounds"].append({
                            "round": 1,
                            "results": new_res,
                            "message": f"Query refined using {len(rel_ids)} relevant and {len(non_rel_ids)} non-relevant selection(s) (round 1)."
                        })
                        st.rerun()

            else:
                latest_round = feedback_rounds[-1]
                latest_round_num = latest_round["round"]

                st.markdown("---")
                st.subheader(f"Refined Results (Round {latest_round_num})")
                st.info(f"✓ {latest_round['message']}")

                for rank, result in enumerate(latest_round["results"], start=1):

                    with st.container(border=True):

                        col1, col2, col3 = st.columns([1, 3.5, 2.5])

                        with col1:
                            st.metric(
                                "Rank",
                                rank
                            )

                            st.metric(
                                "Cosine Score",
                                f"{result['score']:.4f}"
                            )

                        with col2:
                            st.markdown(
                                f"### {result['title']}"
                            )

                            st.write(
                                f"**Document ID:** "
                                f"{result['doc_id']}"
                            )

                            st.write(
                                f"**Category:** "
                                f"{result['category']}"
                            )

                        with col3:
                            st.write("**Relevance Feedback:**")
                            st.radio(
                                "Feedback",
                                ["No opinion", "Relevant", "Not relevant"],
                                horizontal=True,
                                key=f"vsm_fb_{result['doc_id']}_r{latest_round_num}",
                                label_visibility="collapsed"
                            )

                btn_col1, btn_col2 = st.columns([2, 5])

                with btn_col1:
                    if st.button("Refine again", type="primary", key=f"btn_refine_{latest_round_num}"):
                        rel_ids = []
                        non_rel_ids = []
                        for r in latest_round["results"]:
                            val = st.session_state.get(f"vsm_fb_{r['doc_id']}_r{latest_round_num}", "No opinion")
                            if val == "Relevant":
                                rel_ids.append(r["doc_id"])
                            elif val == "Not relevant":
                                non_rel_ids.append(r["doc_id"])

                        if not rel_ids and not non_rel_ids:
                            st.warning("Please mark at least one document as Relevant or Not relevant.")
                        else:
                            next_round = latest_round_num + 1
                            new_res, new_vec, msg = apply_relevance_feedback(
                                current_query_vector=st.session_state["vsm_current_vector"],
                                relevant_doc_ids=rel_ids,
                                non_relevant_doc_ids=non_rel_ids,
                                inverted_index=inverted_index,
                                metadata=metadata,
                                original_results=latest_round["results"]
                            )
                            st.session_state["vsm_current_vector"] = new_vec
                            st.session_state["vsm_feedback_rounds"].append({
                                "round": next_round,
                                "results": new_res,
                                "message": f"Query refined using {len(rel_ids)} relevant and {len(non_rel_ids)} non-relevant selection(s) (round {next_round})."
                            })
                            st.rerun()

                with btn_col2:
                    if st.button("Reset feedback", key=f"btn_reset_{latest_round_num}"):
                        keys_to_clear = [
                            k for k in st.session_state if k.startswith("vsm_fb_")
                        ]
                        for k in keys_to_clear:
                            del st.session_state[k]
                        st.session_state["vsm_feedback_rounds"] = []
                        st.session_state["vsm_current_vector"] = build_query_vector(
                            st.session_state["vsm_active_query"],
                            inverted_index
                        )
                        st.rerun()


elif mode == "Exact Phrase Search":

    st.header("Exact Phrase Search")

    phrase = st.text_input(
        "Enter an exact phrase",
        placeholder="Example: cotton shirt"
    )

    st.info(
        "The phrase must occur using consecutive term positions."
    )

    if st.button("Search Phrase", type="primary"):

        if not phrase.strip():
            st.warning("Please enter a phrase.")
        else:
            results = exact_phrase_search(
                phrase,
                positional_index,
                metadata
            )

            st.subheader("Matching Documents")

            if not results:
                st.error("No exact phrase matches found.")
            else:
                for rank, result in enumerate(results, start=1):

                    with st.container(border=True):

                        st.markdown(
                            f"### {rank}. {result['title']}"
                        )

                        st.write(
                            f"**Document ID:** {result['doc_id']}"
                        )

                        st.write(
                            f"**Category:** {result['category']}"
                        )

                        st.success(
                            "Matching positions found."
                        )

                        st.json(result["matches"])


elif mode == "Ordered Proximity Search":

    st.header("Ordered Proximity Search")

    query = st.text_input(
        "Enter proximity query",
        placeholder="Example: cotton WITHIN/3 shirt"
    )

    st.info(
        "Format: first_term WITHIN/k second_term"
    )

    if st.button("Search Proximity", type="primary"):

        if not query.strip():
            st.warning("Please enter a proximity query.")
        else:

            parsed = parse_proximity_query(query)

            if parsed is None:
                st.error(
                    "Invalid format. Example: "
                    "cotton WITHIN/3 shirt"
                )
            else:

                first_term, second_term, k = parsed

                results = ordered_proximity_search(
                    first_term,
                    second_term,
                    k,
                    positional_index,
                    metadata
                )

                st.subheader(
                    f"Results for {first_term} WITHIN/{k} {second_term}"
                )

                if not results:
                    st.error(
                        "No ordered proximity matches found."
                    )
                else:

                    for rank, result in enumerate(results, start=1):

                        with st.container(border=True):

                            st.markdown(
                                f"### {rank}. {result['title']}"
                            )

                            st.write(
                                f"**Document ID:** {result['doc_id']}"
                            )

                            st.write(
                                f"**Category:** {result['category']}"
                            )

                            st.write(
                                "**Matching Positions:**"
                            )

                            st.json(result["matches"])



elif mode == "Smart Clothing Search (Novelty)":
    st.header("Explainable Smart Clothing Retrieval")
    st.info("Novelty: clothing synonym expansion, weighted relevance ranking, and explanations.")
    query = st.text_input("Enter a smart clothing query", placeholder="Example: black shirt")
    if st.button("Search Smart", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            results = smart_rank_documents(query, inverted_index, metadata)
            if not results:
                st.error("No matching documents found.")
            else:
                for rank, result in enumerate(results, 1):
                    with st.container(border=True):
                        st.markdown(f"### {rank}. {result['title']}")
                        st.write(f"**Document ID:** {result['doc_id']}  |  **Category:** {result['category']}")
                        st.metric("Smart relevance score", f"{result['score']:.4f}")
                        st.write("**Why this result matched:**")
                        for reason in result["explanation"]:
                            st.write("✓ " + reason)

st.sidebar.markdown("---")

st.sidebar.write(
    f"Unique indexed terms: {len(inverted_index)}"
)

st.sidebar.write(
    f"Documents: {len(metadata)}"
)

st.sidebar.markdown(
    """
    **Assignment Features Covered**

    ✓ Part A: Pre-processing  
    ✓ Part B: VSM Ranking  
    ✓ Part C: Positional Index  
    ✓ Part D: User Interface  
    ✓ Part E: Testing
    """
)