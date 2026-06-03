"""
Assignment 3 Group:
Qian Ying Wong, 49411619
"""

import json
from collections import defaultdict
from pathlib import Path

from parser import parse_document
from text_processing import stem_tokens, tokenize_text
from similarity import compute_simhash, hamming_distance, stable_hash_64


NEAR_DUPLICATE_THRESHOLD = 1
# the smaller the threshold, less documents considered as near duplicates
# after testing a few values, seems like 1 makes the most sense for my simhash implementation

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

def get_term_positions(text):
    """
    Returns a map from each stemmed token to the positions where it appears in the document's plain text.
    """
    positions = defaultdict(list)

    for position, token in enumerate(stem_tokens(tokenize_text(text))):
        positions[token].append(position)

    return positions

# helpers for duplicate detection:
def get_similarity_tokens(text):
    return list(stem_tokens(tokenize_text(text)))

def find_near_duplicate(simhash, kept_simhashes, threshold=NEAR_DUPLICATE_THRESHOLD):
    for doc_id, kept_simhash in kept_simhashes:
        if hamming_distance(simhash, kept_simhash) <= threshold:
            return doc_id
    return None


def get_bigrams(tokens): # decided to just do 2-grams instead of 3
    """
    Returns adjacent 2-token phrases from a token list.
    Eg: ["the", "cat", "sat"] -> ["the cat", "cat sat"]
    """
    bigrams = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        bigrams.append(bigram)

    return bigrams


def build_index(corpus_dir):
    """
    Builds an inverted index and document map from the corpus.
    """
    inverted_index = defaultdict(list)
    bigram_index = defaultdict(list)
    doc_map = {}
    kept_simhashes = []
    documents_seen = 0
    near_duplicates_removed = 0

    for file_path in iter_corpus_files(corpus_dir):
        documents_seen += 1
        parsed_document = parse_document(file_path)
        term_positions = get_term_positions(parsed_document["text"])
        similarity_tokens = get_similarity_tokens(parsed_document["text"])
        document_bigrams = get_bigrams(similarity_tokens)
        bigram_counts = defaultdict(int)
        for bigram in document_bigrams:
            bigram_counts[bigram] += 1
        if not similarity_tokens:
            continue
        document_simhash = compute_simhash(similarity_tokens)
        near_duplicate_doc_id = find_near_duplicate(document_simhash, kept_simhashes)

        if near_duplicate_doc_id is not None:
            near_duplicates_removed += 1
            continue
        doc_id = len(doc_map)
        kept_simhashes.append((doc_id, document_simhash))
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
                "positions": term_positions.get(token, []),
            })

        for bigram, frequency in bigram_counts.items():
            bigram_index[bigram].append({
                "doc_id": doc_id,
                "tf": frequency,
            })

    duplicate_stats = {
        "documents_seen": documents_seen,
        "documents_indexed": len(doc_map),
        "near_duplicates_removed": near_duplicates_removed,
    }

    return dict(inverted_index), dict(bigram_index), doc_map, duplicate_stats


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


def save_index(inverted_index, bigram_index, doc_map, output_dir, duplicate_stats=None):
    """
    Saves index artifacts and stats to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inverted_index_path = output_path / "inverted_index.json"
    bigram_index_path = output_path / "bigram_index.json"
    doc_map_path = output_path / "doc_map.json"
    stats_path = output_path / "stats.json"

    save_json(inverted_index, inverted_index_path)
    save_json(bigram_index, bigram_index_path)
    save_json(doc_map, doc_map_path)

    stats = {
        "documents": len(doc_map),
        "unique_tokens": len(inverted_index),
        "unique_bigrams": len(bigram_index),
        "index_size_bytes": 0,
    }
    if duplicate_stats:
        stats.update(duplicate_stats)
    save_json(stats, stats_path)
    stats["index_size_bytes"] = get_directory_size(output_path)
    save_json(stats, stats_path)

    return stats


def main():
    inverted_index, bigram_index, doc_map, duplicate_stats = build_index("ANALYST")
    stats = save_index(inverted_index, bigram_index, doc_map, "index_data", duplicate_stats)

    print(f"No. Documents seen: {stats['documents_seen']}\n"
        f"No. Documents indexed: {stats['documents_indexed']}\n"
        f"No. Near or exact duplicates removed: {stats['near_duplicates_removed']}\n"
        f"No. Unique tokens: {stats['unique_tokens']}\n"
        f"Index size (KB): {stats['index_size_bytes']}")


if __name__ == "__main__":
    main()
