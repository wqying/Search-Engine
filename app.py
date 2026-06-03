import time
from flask import Flask, render_template, request

from search import load_search_data, search


INDEX_DIR = "index_data"

app = Flask(__name__)

print("Loading search index...")
inverted_index, bigram_index, doc_map = load_search_data(INDEX_DIR)
print("Search index loaded.")


@app.route("/")
def home():
    query = request.args.get("q", "").strip()
    results = []
    elapsed_time = None

    if query:
        start_time = time.perf_counter()
        results = search(query, inverted_index, bigram_index, doc_map, top_k=10)
        elapsed_time = time.perf_counter() - start_time

    return render_template(
        "index.html",
        query=query,
        results=results,
        elapsed_time=elapsed_time,
    )


if __name__ == "__main__":
    app.run(debug=True)