"""
Assignment 3 (Information Analyst Track) Group Member(s):
Qian Ying Wong, 49411619
"""

import json
import warnings
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning


TAG_WEIGHTS = {
    "title": 5,
    "h1": 4,
    "h2": 3,
    "h3": 3,
    "b": 2,
    "strong": 2,
}


def get_tag_weight(tag_name):
    """
    Returns the importance weight for an HTML tag.
    Tags that are not specially weighted count as normal text.
    """
    return TAG_WEIGHTS.get(tag_name.lower(), 1)


def parse_document(file_path):
    """
    Reads one corpus JSON file, parses its HTML content, and returns the URL
    plus weighted text sections for the indexer.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        document = json.load(file)
    
    url = document.get("url", "")
    html = document.get("content", "")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning) # supress warnings
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()  # ignores non-text content

    sections = []

    normal_text = soup.get_text(" ", strip=True)
    if normal_text:  # checks if page actually has text
        sections.append((normal_text, get_tag_weight("body")))

    for tag_name in TAG_WEIGHTS:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(" ", strip=True)
            if text:
                sections.append((text, get_tag_weight(tag_name)))

    links = []

    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        anchor_text = anchor.get_text(" ", strip=True)

        if href and anchor_text:
            links.append({
                "href": href,
                "text": anchor_text,
            })

    return {
        "url": url,
        "text": normal_text,  # specifically used for duplicate detection
        "sections": sections,  # used for weighted indexing
        "links": links,  # used for anchor text for Pagerank graph
    }
