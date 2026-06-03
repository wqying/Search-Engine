"""
Assignment 3 (Information Analyst Track) Group Member(s):
Qian Ying Wong, 49411619
"""

import json
from collections import defaultdict
from pathlib import Path

from parser import parse_document
from text_processing import stem_tokens, tokenize_text
from similarity import compute_simhash, hamming_distance
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse


NEAR_DUPLICATE_THRESHOLD = 2
# the smaller the threshold, less documents considered as near duplicates
# after testing a few values, seems like 1 or 2 makes the most sense for my simhash implementation


def normalize_url(url):
    """
    Removes fragments
    """
    clean_url, _ = urldefrag(url.strip())
    parts = urlparse(clean_url)

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"

    for suffix in ("/index.html", "/index.htm", "/index.shtml", "/index.php"):
        if path.lower().endswith(suffix):
            path = path[:-len(suffix)] + "/"

    return urlunparse((scheme, netloc, path, "", parts.query, ""))

def normalize_link(source_url, href):
    return normalize_url(urljoin(source_url, href))


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


def compute_pagerank(link_graph, total_documents, damping=0.85, iterations=30):
    """
    PageRank logic and formula
    """
    ranks = {doc_id: 1 / total_documents for doc_id in range(total_documents)}

    for _ in range(iterations):
        new_ranks = {
            doc_id: (1 - damping) / total_documents
            for doc_id in range(total_documents)
        }

        dangling_rank = sum(
            ranks[doc_id]
            for doc_id in range(total_documents)
            if not link_graph.get(doc_id)
        )

        for doc_id in range(total_documents):
            new_ranks[doc_id] += damping * dangling_rank / total_documents

        for source_doc_id, targets in link_graph.items():
            if not targets:
                continue

            share = damping * ranks[source_doc_id] / len(targets)

            for target_doc_id in targets:
                new_ranks[target_doc_id] += share

        ranks = new_ranks

    return {str(doc_id): rank for doc_id, rank in ranks.items()}

def collect_kept_documents(corpus_dir):
    """
    First pass over the corpus:
    Parses documents, removes near duplicates, assigns doc_ids, and builds
    URL lookup structures needed for link graph and anchor text indexing.
    """
    kept_documents = []
    doc_map = {}
    url_to_doc_id = {}
    kept_simhashes = []

    documents_seen = 0
    near_duplicates_removed = 0

    for file_path in iter_corpus_files(corpus_dir):
        documents_seen += 1

        parsed_document = parse_document(file_path)
        similarity_tokens = get_similarity_tokens(parsed_document["text"])

        if not similarity_tokens:
            continue

        document_simhash = compute_simhash(similarity_tokens)
        near_duplicate_doc_id = find_near_duplicate(document_simhash, kept_simhashes)

        if near_duplicate_doc_id is not None:
            near_duplicates_removed += 1
            continue

        doc_id = len(kept_documents)
        normalized_url = normalize_url(parsed_document["url"])

        kept_simhashes.append((doc_id, document_simhash))
        url_to_doc_id[normalized_url] = doc_id

        doc_map[doc_id] = {
            "url": parsed_document["url"],
            "normalized_url": normalized_url,
            "path": str(file_path),
        }

        kept_documents.append({
            "doc_id": doc_id,
            "parsed_document": parsed_document,
            "tokens": similarity_tokens,
            "normalized_url": normalized_url,
        })

    duplicate_stats = {
        "documents_seen": documents_seen,
        "documents_indexed": len(doc_map),
        "near_duplicates_removed": near_duplicates_removed,
    }

    return kept_documents, doc_map, url_to_doc_id, duplicate_stats


def build_text_indexes(kept_documents):
    """
    Builds the unigram inverted index and bigram index for kept documents.
    """
    inverted_index = defaultdict(list)
    bigram_index = defaultdict(list)

    for document in kept_documents:
        doc_id = document["doc_id"]
        parsed_document = document["parsed_document"]

        term_positions = get_term_positions(parsed_document["text"])
        raw_counts, weighted_counts = count_document_terms(parsed_document["sections"])

        for token, term_frequency in raw_counts.items():
            inverted_index[token].append({
                "doc_id": doc_id,
                "tf": term_frequency,
                "weighted_tf": weighted_counts[token],
                "positions": term_positions.get(token, []),
            })

        bigram_counts = defaultdict(int)
        for bigram in get_bigrams(document["tokens"]):
            bigram_counts[bigram] += 1

        for bigram, frequency in bigram_counts.items():
            bigram_index[bigram].append({
                "doc_id": doc_id,
                "tf": frequency,
            })

    return dict(inverted_index), dict(bigram_index)


def build_link_graph(kept_documents, url_to_doc_id):
    """
    Builds a graph where each doc_id points to the doc_ids it links to.
    """
    link_graph = {
        document["doc_id"]: set()
        for document in kept_documents
    }

    for document in kept_documents:
        source_doc_id = document["doc_id"]
        source_url = document["normalized_url"]
        parsed_document = document["parsed_document"]

        for link in parsed_document.get("links", []):
            target_url = normalize_link(source_url, link["href"])
            target_doc_id = url_to_doc_id.get(target_url)

            if target_doc_id is None:
                continue

            if target_doc_id == source_doc_id:
                continue

            link_graph[source_doc_id].add(target_doc_id)

    return link_graph


def build_anchor_index(kept_documents, url_to_doc_id):
    """
    Builds an anchor-text index for target pages.
    """
    anchor_counts = defaultdict(lambda: defaultdict(int))

    for document in kept_documents:
        source_url = document["normalized_url"]
        parsed_document = document["parsed_document"]

        for link in parsed_document.get("links", []):
            target_url = normalize_link(source_url, link["href"])
            target_doc_id = url_to_doc_id.get(target_url)

            if target_doc_id is None:
                continue

            anchor_text = link.get("text", "")

            for token in stem_tokens(tokenize_text(anchor_text)):
                anchor_counts[target_doc_id][token] += 1

    anchor_index = defaultdict(list)

    for target_doc_id, token_counts in anchor_counts.items():
        for token, frequency in token_counts.items():
            anchor_index[token].append({
                "doc_id": target_doc_id,
                "tf": frequency,
            })

    return dict(anchor_index)


def build_index(corpus_dir):
    """
    Builds index artifacts from the corpus using a two-pass structure.
    """
    kept_documents, doc_map, url_to_doc_id, duplicate_stats = collect_kept_documents(corpus_dir)
    inverted_index, bigram_index = build_text_indexes(kept_documents)
    link_graph = build_link_graph(kept_documents, url_to_doc_id)
    pagerank = compute_pagerank(link_graph, len(doc_map))
    link_graph_json = {
        str(doc_id): sorted(target_doc_ids)
        for doc_id, target_doc_ids in link_graph.items()
    }
    anchor_index = build_anchor_index(kept_documents, url_to_doc_id)

    return (
        inverted_index,
        bigram_index,
        anchor_index,
        link_graph_json,
        pagerank,
        doc_map,
        duplicate_stats,
    )


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


def save_index(inverted_index, bigram_index, anchor_index, link_graph, pagerank, doc_map, output_dir, duplicate_stats=None):
    """
    Saves index artifacts and stats to disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inverted_index_path = output_path / "inverted_index.json"
    bigram_index_path = output_path / "bigram_index.json"
    anchor_index_path = output_path / "anchor_index.json"
    link_graph_path = output_path / "link_graph.json"
    pagerank_path = output_path / "pagerank.json"
    doc_map_path = output_path / "doc_map.json"
    stats_path = output_path / "stats.json"
    

    save_json(inverted_index, inverted_index_path)
    save_json(bigram_index, bigram_index_path)
    save_json(anchor_index, anchor_index_path)
    save_json(link_graph, link_graph_path)
    save_json(pagerank, pagerank_path)  
    save_json(doc_map, doc_map_path)

    stats = {
        "documents": len(doc_map),
        "unique_tokens": len(inverted_index),
        "unique_bigrams": len(bigram_index),
        "index_size_bytes": 0,
        "unique_anchor_tokens": len(anchor_index),
        "links": sum(len(targets) for targets in link_graph.values()),
    }

    if duplicate_stats:
        stats.update(duplicate_stats)
    save_json(stats, stats_path)
    stats["index_size_bytes"] = get_directory_size(output_path)
    save_json(stats, stats_path)

    return stats


def main():
    inverted_index, bigram_index, anchor_index, link_graph, pagerank, doc_map, duplicate_stats = build_index("ANALYST")
    stats = save_index(inverted_index, bigram_index, anchor_index, link_graph, pagerank, doc_map, "index_data", duplicate_stats)

    print(f"No. Documents seen: {stats['documents_seen']}\n"
        f"No. Documents indexed: {stats['documents_indexed']}\n"
        f"No. Near or exact duplicates removed: {stats['near_duplicates_removed']}\n"
        f"No. Unique tokens: {stats['unique_tokens']}\n"
        f"No. Unique bigrams: {stats['unique_bigrams']}\n"
        f"No. Unique anchor tokens: {stats['unique_anchor_tokens']}\n"
        f"No. Links in graph: {stats['links']}\n"
        f"Index size (KB): {stats['index_size_bytes']}")


if __name__ == "__main__":
    main()
