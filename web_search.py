"""
web_search.py - Zero-Dependency Web Search Tool for Agentic AI
Uses standard library urllib to search DuckDuckGo and Wikipedia without extra frameworks.
"""

import html
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List

WEB_SEARCH_TOOL_DECLARATION = {
    "name": "web_search",
    "description": "Searches the web for up-to-date information, facts, definitions, current events, or general knowledge.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The search query string to look up on the web."
            }
        },
        "required": ["query"]
    }
}


def _search_duckduckgo_api(query: str) -> List[Dict[str, str]]:
    """Fetches instant answers from DuckDuckGo API."""
    results = []
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        })
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgenticAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))

            abstract = data.get("AbstractText", "").strip()
            source = data.get("AbstractSource", "DuckDuckGo")
            abstract_url = data.get("AbstractURL", "")

            if abstract:
                results.append({
                    "title": data.get("Heading", query),
                    "snippet": abstract,
                    "source": source,
                    "url": abstract_url
                })

            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "snippet": topic["Text"],
                        "source": "DuckDuckGo Related",
                        "url": topic.get("FirstURL", "")
                    })
    except Exception:
        pass
    return results


def _search_wikipedia_fulltext(query: str) -> List[Dict[str, str]]:
    """Searches Wikipedia full-text and fetches intro extracts."""
    results = []
    try:
        # Step 1: Search Wikipedia for relevant article titles
        search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "utf8": "1",
            "format": "json"
        })
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgenticAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
            search_items = data.get("query", {}).get("search", [])

        if not search_items:
            return results

        # Top matching page titles
        top_titles = [item["title"] for item in search_items[:2]]
        
        # Step 2: Fetch clean plain-text intro extracts for the top titles
        extract_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": "|".join(top_titles),
            "format": "json"
        })
        req2 = urllib.request.Request(
            extract_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgenticAI/1.0"}
        )
        with urllib.request.urlopen(req2, timeout=6) as response2:
            data2 = json.loads(response2.read().decode("utf-8"))
            pages = data2.get("query", {}).get("pages", {})

        for pid, page in pages.items():
            extract = page.get("extract", "").strip()
            title = page.get("title", "")
            if extract and "may refer to:" not in extract:
                # Truncate clean extract to reasonable snippet
                snippet = extract[:400] + ("..." if len(extract) > 400 else "")
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "source": "Wikipedia",
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                })
    except Exception:
        pass
    return results


def web_search(query: str) -> Dict[str, Any]:
    """
    Executes a web search query and returns structured results.
    """
    clean_query = query.strip()
    if not clean_query:
        return {"status": "error", "message": "Search query cannot be empty."}

    # 1. Try DuckDuckGo Instant Answer
    results = _search_duckduckgo_api(clean_query)

    # 2. If no instant answer or sparse results, search Wikipedia full-text
    if len(results) < 2:
        wiki_results = _search_wikipedia_fulltext(clean_query)
        results.extend(wiki_results)

    if not results:
        return {
            "status": "not_found",
            "query": clean_query,
            "message": f"No web search results found for '{clean_query}'."
        }

    return {
        "status": "success",
        "query": clean_query,
        "results": results[:4]
    }


if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "who founded Python programming language"
    res = web_search(test_q)
    print(json.dumps(res, indent=2))
