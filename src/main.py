from pathlib import Path

from src.indexer import build_and_save
from src.retrieval import search as vector_search
from src.positional_search import (
    exact_phrase_search,
    proximity_search
)


INDEX_DIR = Path(__file__).resolve().parent.parent / "data"


def ensure_indexes_exist():
    """
    Check whether all required index files exist.
    Build them automatically if they are missing.
    """

    required_files = [
        INDEX_DIR / "inverted_index.json",
        INDEX_DIR / "positional_index.json",
        INDEX_DIR / "document_metadata.json",
    ]

    if not all(file.exists() for file in required_files):
        print("Indexes not found. Building indexes...\n")

        build_and_save()

        print("\nIndexes built successfully.\n")


def display_results(results):
    """
    Display search results in a readable format.
    """

    if not results:
        print("\nNo matching documents found.")
        return

    print(f"\nFound {len(results)} result(s):\n")

    for result in results:
        doc_id = result["doc_id"]

        print(f"Document ID: {doc_id}")

        if "score" in result:
            print(f"Cosine Score: {result['score']:.4f}")

        if "title" in result:
            print(f"Title: {result['title']}")

        if "category" in result:
            print(f"Category: {result['category']}")

        if "matches" in result:
            print(f"Matches: {result['matches']}")

        print("-" * 60)


def main():
    ensure_indexes_exist()

    while True:
        print("\n===== CLOTHING INFORMATION RETRIEVAL SYSTEM =====")
        print("1. Free-Text Vector Space Search")
        print("2. Exact Phrase Search")
        print("3. Proximity Search")
        print("4. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            query = input("Enter free-text query: ").strip()

            if not query:
                print("Query cannot be empty.")
                continue

            results = vector_search(query)
            display_results(results)

        elif choice == "2":
            phrase = input("Enter exact phrase: ").strip()

            if not phrase:
                print("Phrase cannot be empty.")
                continue

            results = exact_phrase_search(phrase)
            display_results(results)

        elif choice == "3":
            query = input(
                "Enter proximity query "
                "(example: cotton WITHIN/3 shirt): "
            ).strip()

            if not query:
                print("Query cannot be empty.")
                continue

            results = proximity_search(query)
            display_results(results)

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()