import sys
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from src.indexer import build_and_save
from src.retrieval import load_indexes, rank_documents, smart_rank_documents
from src.positional_search import (
    exact_phrase_search,
    ordered_proximity_search,
    parse_proximity_query
)


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
        else:
            results = rank_documents(
                query,
                inverted_index,
                metadata
            )

            st.subheader("Top 10 Ranked Results")

            if not results:
                st.error(
                    "No matching documents found. "
                    "Try another query."
                )
            else:
                for rank, result in enumerate(results, start=1):

                    with st.container(border=True):

                        col1, col2 = st.columns([1, 5])

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