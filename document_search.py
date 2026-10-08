"""
document_search.py - Zero-Dependency Local Document Search Tool for Agentic AI
Searches local documents (.txt, .md, .json, .csv) in the 'documents/' folder
using Python's standard library. Zero external frameworks.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DOCS_DIR = Path(__file__).parent / "documents"

DOCUMENT_SEARCH_TOOL_DECLARATION = {
    "name": "document_search",
    "description": "Searches local documents, text files, and notes in the documents directory for matching keywords, topics, or content.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query": {
                "type": "STRING",
                "description": "The keyword, phrase, or topic to search for in local documents."
            },
            "filename_filter": {
                "type": "STRING",
                "description": "Optional specific filename to search within (e.g. 'company_policy.txt' or '*.md'). If omitted, searches all documents."
            }
        },
        "required": ["query"]
    }
}


def document_search(query: str, filename_filter: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Searches local documents for lines and paragraphs matching the query.
    """
    clean_query = (query or "").strip().lower()
    if not clean_query:
        return {"status": "error", "message": "Search query cannot be empty."}

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "status": "not_found",
            "query": query,
            "message": f"Documents folder '{DOCS_DIR}' was empty or not found."
        }

    # Find matching files in documents directory
    search_pattern = filename_filter if filename_filter and "*" in filename_filter else "*.*"
    candidate_files = list(DOCS_DIR.glob(search_pattern))

    if filename_filter and "*" not in filename_filter:
        candidate_files = [f for f in candidate_files if f.name.lower() == filename_filter.lower()]

    # Supported text-based extensions
    valid_extensions = {".txt", ".md", ".json", ".csv", ".log", ".py"}
    text_files = [f for f in candidate_files if f.is_file() and f.suffix.lower() in valid_extensions]

    if not text_files:
        return {
            "status": "not_found",
            "query": query,
            "message": f"No searchable text documents found in '{DOCS_DIR}' matching filter '{filename_filter}'."
        }

    query_tokens = set(re.findall(r"\w+", clean_query))
    matches = []

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line_idx, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                # Check for exact phrase match or token matches
                if clean_query in line_lower:
                    score = 1.0
                else:
                    line_tokens = set(re.findall(r"\w+", line_lower))
                    overlap = query_tokens.intersection(line_tokens)
                    score = len(overlap) / len(query_tokens) if query_tokens else 0.0

                if score >= 0.5:
                    matches.append({
                        "file": file_path.name,
                        "line_number": line_idx,
                        "content": line.strip(),
                        "relevance_score": round(score, 2)
                    })
        except Exception:
            continue

    # Sort matches by relevance score descending
    matches.sort(key=lambda x: x["relevance_score"], reverse=True)

    if not matches:
        return {
            "status": "not_found",
            "query": query,
            "documents_searched": len(text_files),
            "message": f"No matches found for '{query}' across {len(text_files)} local documents."
        }

    return {
        "status": "success",
        "query": query,
        "total_matches": len(matches),
        "results": matches[:5]
    }


if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "vacation days"
    res = document_search(test_q)
    print(json.dumps(res, indent=2))
