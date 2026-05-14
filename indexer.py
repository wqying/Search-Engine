import json
import sys
from collections import defaultdict
from pathlib import Path

from parser import parse_document
from text_processing import stem_tokens, tokenize_text


def iter_corpus_files(corpus_dir):
    """
    Yields JSON files in the corpus directory in a stable order.
    """
    corpus_path = Path(corpus_dir)
    yield from sorted(corpus_path.rglob("*.json"))


def count_document_terms(sections):
    """
    Counts raw and weighted term frequencies for one parsed document.
    """
    raw_counts = defaultdict(int)
    weighted_counts = defaultdict(int)

    for text, weight in sections:
        tokens = stem_tokens(tokenize_text(text))  # stem first then count

        for token in tokens:
            raw_counts[token] += 1
            weighted_counts[token] += weight

    return raw_counts, weighted_counts


def build_index(corpus_dir):
    """
    Builds an inverted index and document map from the corpus.
    """
    inverted_index = defaultdict(list)
    doc_map = {}

    for doc_id, file_path in enumerate(iter_corpus_files(corpus_dir)):
        parsed_document = parse_document(file_path)
        raw_counts, weighted_counts = count_document_terms(parsed_document["sections"])

        doc_map[doc_id] = {
            "url": parsed_document["url"],
            "path": str(file_path),
        }

        for token, term_frequency in raw_counts.items():
            inverted_index[token].append({
                "doc_id": doc_id,
                "tf": term_frequency,
                "weighted_tf": weighted_counts[token],
            })

    return dict(inverted_index), doc_map


def save_json(data, file_path):
    """
    Writes JSON data to disk.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_directory_size(directory):
    """
    Returns the total size of files in a directory.
    """
    total_size = 0

    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return total_size // 1024  # convert bytes to KB


def save_index(inverted_index, doc_map, output_dir):
    """
    Saves index artifacts and stats to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inverted_index_path = output_path / "inverted_index.json"
    doc_map_path = output_path / "doc_map.json"
    stats_path = output_path / "stats.json"

    save_json(inverted_index, inverted_index_path)
    save_json(doc_map, doc_map_path)

    stats = {
        "documents": len(doc_map),
        "unique_tokens": len(inverted_index),
        "index_size_bytes": 0,
    }
    save_json(stats, stats_path)
    stats["index_size_bytes"] = get_directory_size(output_path)
    save_json(stats, stats_path)

    return stats


def main():
    inverted_index, doc_map = build_index("ANALYST")
    stats = save_index(inverted_index, doc_map, "index_data")

    print(f"No. Documents indexed: {stats['documents']}\n"
          f"No. Unique tokens: {stats['unique_tokens']}\n"
          f"Index size (KB): {stats['index_size_bytes']}")


if __name__ == "__main__":
    main()
