"""
Assignment 3 (Information Analyst Track) Group Member(s):
Qian Ying Wong, 49411619
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from indexer import get_bigrams
from text_processing import stem_tokens, tokenize_text


QUERY_STOP_WORDS = { # ignored during ranking bc they've been messing with results
    "a", "an", "and", "are", "as", "at",
    "be", "by", "for", "from", "in", "is",
    "of", "on", "or", "that", "the", "to",
    "with",
}
PAGERANK_WEIGHT = 10


def load_search_data(index_dir):
    """
    Loads the inverted index and document map once at startup.
    """
    index_path = Path(index_dir)

    with open(index_path / "inverted_index.json", "r", encoding="utf-8") as file:
        inverted_index = json.load(file)

    with open(index_path / "bigram_index.json", "r", encoding="utf-8") as file:
        bigram_index = json.load(file)

    with open(index_path / "pagerank.json", "r", encoding="utf-8") as file:
        pagerank = json.load(file)

    with open(index_path / "doc_map.json", "r", encoding="utf-8") as file:
        doc_map = json.load(file)

    return inverted_index, bigram_index, pagerank, doc_map


def normalize_query(query):
    """
    Tokenizes and stems a query using the same processing as the indexer,
    but also removes stop words that were found to be hurting ranking performance.
    """
    filtered_tokens = []
    for token in tokenize_text(query):
        if token not in QUERY_STOP_WORDS:
            filtered_tokens.append(token)
    stemmed_tokens = stem_tokens(filtered_tokens)
    return list(dict.fromkeys(stemmed_tokens))


def get_minimum_query_span(query_tokens, document_positions):
    """
    Returns the smallest word-position span containing all query tokens.
    """
    if len(query_tokens) < 2:
        return None

    for token in query_tokens:
        if token not in document_positions or not document_positions[token]:
            return None

    occurrences = []
    for token in query_tokens:
        for position in document_positions[token]:
            occurrences.append((position, token))

    occurrences.sort()

    required_terms = len(query_tokens)
    window_counts = defaultdict(int)
    terms_in_window = 0
    left = 0
    best_span = None

    for right, (right_position, right_token) in enumerate(occurrences):
        if window_counts[right_token] == 0:
            terms_in_window += 1
        window_counts[right_token] += 1

        while terms_in_window == required_terms:
            left_position, left_token = occurrences[left]
            current_span = right_position - left_position

            if best_span is None or current_span < best_span:
                best_span = current_span

            window_counts[left_token] -= 1
            if window_counts[left_token] == 0:
                terms_in_window -= 1

            left += 1

    return best_span


def search(query, inverted_index, bigram_index, pagerank, doc_map, top_k=10, offset=0):
    """
    Runs an OR query and returns the top results ranked by weighted tf-idf, proximity, and bigram matches.
    formula: score = (weighted tf-idf + proximity * 2 + bigram * 4) * query_coverage + normalized_pagerank * 10
    """
    query_tokens = normalize_query(query)
    query_bigrams = get_bigrams(query_tokens)

    if not query_tokens:
        return []

    postings_by_token = []
    candidate_doc_ids = set()

    for token in query_tokens:
        postings = inverted_index.get(token)
        if postings is None:
            continue
        postings_by_token.append((token, postings))
        for posting in postings:
            candidate_doc_ids.add(posting["doc_id"]) # union of all documents with at least one query token

    if not candidate_doc_ids:
        return []

    total_documents = len(doc_map)
    scores = {doc_id: 0.0 for doc_id in candidate_doc_ids}
    term_frequencies = {doc_id: 0 for doc_id in candidate_doc_ids}
    matched_terms = {doc_id: 0 for doc_id in candidate_doc_ids} # tie breaker
    positions_by_doc = {doc_id: {} for doc_id in candidate_doc_ids}

    proximity_scores = {doc_id: 0.0 for doc_id in candidate_doc_ids}
    bigram_scores = {doc_id: 0.0 for doc_id in candidate_doc_ids}

    # tf-idf
    for token, postings in postings_by_token:
        document_frequency = len(postings)
        inverse_document_frequency = math.log(
            (total_documents + 1) / (document_frequency + 1)
            ) + 1 # normalized idf fomula
        # self note: rare terms have higher idf score

        for posting in postings:
            doc_id = posting["doc_id"]
            if doc_id in candidate_doc_ids:  # document score is updated here
                scores[doc_id] += posting["weighted_tf"] * inverse_document_frequency
                term_frequencies[doc_id] += posting["tf"]
                matched_terms[doc_id] += 1
                positions_by_doc[doc_id][token] = posting.get("positions", [])

    # position and proximity
    for doc_id in candidate_doc_ids:
        minimum_span = get_minimum_query_span(query_tokens, positions_by_doc[doc_id])

        if minimum_span is not None:
            # adjacent query terms have bigger bonus scores
            proximity_scores[doc_id] = len(query_tokens) / (minimum_span + 1)
            scores[doc_id] += proximity_scores[doc_id] * 2.0 # proximity gets big boost, but less than bigram

    # 2-gram
    for bigram in query_bigrams:
        postings = bigram_index.get(bigram)

        if postings is None:
            continue

        for posting in postings:
            doc_id = posting["doc_id"]

            if doc_id in candidate_doc_ids:
                bigram_scores[doc_id] += posting["tf"]
                scores[doc_id] += posting["tf"] * 4.0  # exactly adjacent gets stronger boost

    # ensure documents that match more of the query are ranked higher than those that only match a small portion
    for doc_id in candidate_doc_ids:
        query_coverage = matched_terms[doc_id] / len(query_tokens)
        scores[doc_id] *= query_coverage

    pagerank_scores = {doc_id: pagerank.get(str(doc_id), 0.0) for doc_id in candidate_doc_ids}

    max_pagerank = max(pagerank_scores.values()) if pagerank_scores else 0.0

    for doc_id in candidate_doc_ids:
        if max_pagerank > 0:
            normalized_pagerank = pagerank_scores[doc_id] / max_pagerank
        else:
            normalized_pagerank = 0.0

        scores[doc_id] += normalized_pagerank * PAGERANK_WEIGHT

    # sort candidate documents best to worst:
    # 1. highest tf-idf score
    # 2. if tied, highest number of matched query terms
    # 3. if tied, highest raw term frequency
    # 4. if tied, lowest doc_id
    ranked_doc_ids = sorted(
        candidate_doc_ids,
        key=lambda doc_id: (
            -scores[doc_id],
            -matched_terms[doc_id],
            -term_frequencies[doc_id],
            doc_id,
        ),
    )

    results = []
    for doc_id in ranked_doc_ids[offset:offset + top_k]:
        document = doc_map[str(doc_id)]
        results.append({
            "doc_id": doc_id,
            "url": document["url"],
            "score": scores[doc_id],
            "tf": term_frequencies[doc_id],
            "matched_terms": matched_terms[doc_id],
            "proximity": proximity_scores[doc_id],
            "bigram": bigram_scores[doc_id],
            "pagerank": pagerank_scores[doc_id],
        })

    return results
