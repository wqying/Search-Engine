import argparse
import json
import time
from pathlib import Path

from text_processing import stem_tokens, tokenize_text


DEFAULT_INDEX_DIR = "index_data"
REQUIRED_QUERIES = [
    "cristina lopes",
    "machine learning",
    "ACM",
    "master of software engineering",
]


def load_search_data(index_dir):
    """
    Loads the inverted index and document map once at startup.
    """
    index_path = Path(index_dir)

    with open(index_path / "inverted_index.json", "r", encoding="utf-8") as file:
        inverted_index = json.load(file)

    with open(index_path / "doc_map.json", "r", encoding="utf-8") as file:
        doc_map = json.load(file)

    return inverted_index, doc_map


def normalize_query(query):
    """
    Tokenizes and stems a query using the same processing as the indexer.
    """
    tokens = stem_tokens(tokenize_text(query))
    return list(dict.fromkeys(tokens))


def search(query, inverted_index, doc_map, top_k=5):
    """
    Runs an AND query and returns the top results ranked by summed weighted_tf.
    """
    query_tokens = normalize_query(query)

    if not query_tokens:
        return []

    postings_by_token = []
    for token in query_tokens:
        postings = inverted_index.get(token)
        if postings is None:
            return []
        postings_by_token.append(postings)

    matching_doc_ids = None
    for postings in postings_by_token:
        doc_ids = {posting["doc_id"] for posting in postings}
        if matching_doc_ids is None:
            matching_doc_ids = doc_ids
        else:
            matching_doc_ids &= doc_ids

    if not matching_doc_ids:
        return []

    scores = {doc_id: 0 for doc_id in matching_doc_ids}
    term_frequencies = {doc_id: 0 for doc_id in matching_doc_ids}

    for postings in postings_by_token:
        for posting in postings:
            doc_id = posting["doc_id"]
            if doc_id in matching_doc_ids:
                scores[doc_id] += posting["weighted_tf"]
                term_frequencies[doc_id] += posting["tf"]

    ranked_doc_ids = sorted(
        matching_doc_ids,
        key=lambda doc_id: (-scores[doc_id], -term_frequencies[doc_id], doc_id),
    )

    results = []
    for doc_id in ranked_doc_ids[:top_k]:
        document = doc_map[str(doc_id)]
        results.append({
            "doc_id": doc_id,
            "url": document["url"],
            "score": scores[doc_id],
            "tf": term_frequencies[doc_id],
        })

    return results


def print_results(query, results, elapsed_time):
    """
    Prints one query's search results.
    """
    print(f"\nQuery: {query}")
    print(f"Returned {len(results)} result(s) in {elapsed_time:.4f} seconds")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['url']}")
        print(f"   score={result['score']} tf={result['tf']}")


def print_required_queries(index_dir):
    """
    Prints the required M2 query results to the terminal.
    """
    inverted_index, doc_map = load_search_data(index_dir)

    for query in REQUIRED_QUERIES:
        start_time = time.perf_counter()
        results = search(query, inverted_index, doc_map)
        elapsed_time = time.perf_counter() - start_time
        print_results(query, results, elapsed_time)


def run_interactive_search(index_dir):
    """
    Runs a text-based search interface.
    """
    print("Loading index...")
    inverted_index, doc_map = load_search_data(index_dir)
    print("Search engine ready. Type a query, or type q to quit.")

    while True:
        query = input("\nSearch> ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break

        start_time = time.perf_counter()
        results = search(query, inverted_index, doc_map)
        elapsed_time = time.perf_counter() - start_time

        print_results(query, results, elapsed_time)


def main():
    arg_parser = argparse.ArgumentParser(description="Search the indexed corpus.")
    arg_parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    arg_parser.add_argument("--required", action="store_true")
    args = arg_parser.parse_args()

    if args.required:
        print_required_queries(args.index_dir)
    else:
        run_interactive_search(args.index_dir)


if __name__ == "__main__":
    main()
