import json
import re
from pathlib import Path

from src.preprocess import normalize_tokens


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "corpus_100.txt"
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


def extract_tag(block, tag):
    """
    Extract content between XML-style tags.

    Handles optional spaces inside tags, for example:
        <DOCID>D001</DOCID>
        <DOCID> D001 </DOCID>
    """

    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"

    match = re.search(
        pattern,
        block,
        re.DOTALL | re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return ""


def parse_corpus():
    """
    Parse corpus_100.txt into a list of documents.
    """

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus file not found at: {CORPUS_PATH}"
        )

    content = CORPUS_PATH.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Corpus size: {len(content)} characters")

    # Handles <DOC>, < DOC >, and different capitalization.
    document_blocks = re.findall(
        r"<\s*DOC\s*>(.*?)<\s*/\s*DOC\s*>",
        content,
        re.DOTALL | re.IGNORECASE
    )

    print(f"Raw DOC blocks found: {len(document_blocks)}")

    documents = []

    for block in document_blocks:
        doc_id = extract_tag(block, "DOCID")
        category = extract_tag(block, "CATEGORY")
        title = extract_tag(block, "TITLE")
        text = extract_tag(block, "TEXT")

        if doc_id:
            documents.append({
                "doc_id": doc_id,
                "category": category,
                "title": title,
                "text": text
            })

    def numeric_doc_id(doc):
        match = re.search(r"\d+", doc["doc_id"])

        if match:
            return int(match.group())

        return float("inf")

    documents.sort(key=numeric_doc_id)

    return documents


def build_indexes(documents):
    """
    Build inverted and positional indexes.
    """

    inverted_index = {}
    positional_index = {}
    metadata = {}

    for document in documents:
        doc_id = document["doc_id"]
        title = document["title"]
        category = document["category"]
        text = document["text"]

        metadata[doc_id] = {
            "title": title,
            "category": category,
            "text": text
        }

        terms = normalize_tokens(
            title + " " + text
        )

        term_positions = {}

        for position, term in enumerate(terms):
            term_positions.setdefault(term, []).append(position)

        for term, positions in term_positions.items():
            tf = len(positions)

            if term not in inverted_index:
                inverted_index[term] = {
                    "df": 0,
                    "postings": []
                }

            inverted_index[term]["df"] += 1

            inverted_index[term]["postings"].append({
                "doc_id": doc_id,
                "tf": tf
            })

            if term not in positional_index:
                positional_index[term] = {
                    "df": 0,
                    "postings": []
                }

            positional_index[term]["df"] += 1

            positional_index[term]["postings"].append({
                "doc_id": doc_id,
                "tf": tf,
                "positions": positions
            })

    return inverted_index, positional_index, metadata


def save_indexes(inverted_index, positional_index, metadata):
    """
    Save generated indexes as JSON files.
    """

    with open(
        DATA_DIR / "inverted_index.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            inverted_index,
            file,
            indent=2,
            ensure_ascii=False
        )

    with open(
        DATA_DIR / "positional_index.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            positional_index,
            file,
            indent=2,
            ensure_ascii=False
        )

    with open(
        DATA_DIR / "document_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )


def build_and_save():
    """
    Main indexing function.
    """

    print("Reading corpus...\n")

    documents = parse_corpus()

    print(f"Documents found: {len(documents)}")

    if len(documents) == 0:
        print("\nERROR: No documents were parsed.")
        print("Please check that corpus_100.txt contains <DOC> blocks.")
        print("Expected format:")
        print("<DOC>")
        print("<DOCID>D001</DOCID>")
        print("<CATEGORY>Shirt</CATEGORY>")
        print("<TITLE>Example title</TITLE>")
        print("<TEXT>Example description</TEXT>")
        print("</DOC>")
        return

    inverted_index, positional_index, metadata = build_indexes(
        documents
    )

    save_indexes(
        inverted_index,
        positional_index,
        metadata
    )

    print("\nIndexes successfully created.")
    print(f"Unique terms: {len(inverted_index)}")
    print("Saved files:")
    print(" - data/inverted_index.json")
    print(" - data/positional_index.json")
    print(" - data/document_metadata.json")


if __name__ == "__main__":
    build_and_save()