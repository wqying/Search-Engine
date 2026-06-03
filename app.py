"""
Assignment 3 (Information Analyst Track) Group Member(s):
Qian Ying Wong, 49411619

Web interface for my search engine. Entrace to my project.
For TAs: install requirements.txt and run `python3 app.py` in the terminal
"""

import time
from flask import Flask, render_template, request
from search import load_search_data, search


INDEX_DIR = "index_data"
RESULTS_PER_PAGE = 10

app = Flask(__name__)

print("Loading search index...")
inverted_index, bigram_index, pagerank, doc_map = load_search_data(INDEX_DIR)
print("Search index loaded.")


@app.route("/")
def home():
    query = request.args.get("q", "").strip()
    page = request.args.get("page", default=1, type=int)
    page = max(page, 1)
    results = []
    elapsed_time = None
    has_next = False

    if query:
        offset = (page - 1) * RESULTS_PER_PAGE
        start_time = time.perf_counter()
        paged_results = search(
            query,
            inverted_index,
            bigram_index,
            pagerank,
            doc_map,
            top_k=RESULTS_PER_PAGE + 1,
            offset=offset,
        )
        elapsed_time = time.perf_counter() - start_time
        has_next = len(paged_results) > RESULTS_PER_PAGE
        results = paged_results[:RESULTS_PER_PAGE]

    return render_template(
        "index.html",
        query=query,
        results=results,
        elapsed_time=elapsed_time,
        page=page,
        has_next=has_next,
        start_rank=(page - 1) * RESULTS_PER_PAGE,
    )


if __name__ == "__main__":
    app.run(debug=True)
