"""
server.py - Zero-Framework Full-Stack Backend for Agentic AI & RAG Studio
Uses Python's standard library `http.server.ThreadingHTTPServer` to serve:
1. Modern Frontend Web UI (HTML/CSS/JS)
2. REST API for ReAct Multi-Tool Agent (/api/chat)
3. REST API for Student Database RAG Agent (/api/rag)
"""

import json
import os
import sys
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configuration
GEMINI_API_KEY = "AQ.Ab8RN6JYtoXDU0h1BYtDraFHAv1b6v69VIb3PMntbf5j0kp5BA" or os.environ.get("GEMINI_API_KEY")
PORT = int(os.environ.get("PORT", 8000))
HOST = "127.0.0.1"
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Import our existing agents
from agent import CalculatorAgent
from rag_agent import StudentRAGAgent

# Shared Agent Instances
print("[Backend] Initializing ReAct Multi-Tool Agent...")
react_agent = CalculatorAgent(api_key=GEMINI_API_KEY, verbose=True)

print("[Backend] Initializing Student Database RAG Agent...")
rag_agent = StudentRAGAgent()


class AgentHTTPHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and REST API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve index.html at root
        if parsed.path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
                return

        # Serve tools schema API
        if parsed.path == "/api/tools":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            tools_info = [
                {"name": "calculate", "description": "Deterministic AST Math Calculator"},
                {"name": "web_search", "description": "Online live web facts and search"},
                {"name": "get_current_datetime", "description": "Live System Date, Time, Day"},
                {"name": "document_search", "description": "Local Document & Policy Search"}
            ]
            self.wfile.write(json.dumps(tools_info).encode("utf-8"))
            return

        # Fallback to serving static files
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)

        # Read JSON body
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            req_json = {}

        # Extract dynamic API key if provided by frontend
        custom_key = (req_json.get("api_key") or self.headers.get("X-Gemini-API-Key") or "").strip()
        effective_key = custom_key if custom_key else GEMINI_API_KEY

        if effective_key:
            react_agent.client.api_key = effective_key
            if hasattr(rag_agent, "llm"):
                rag_agent.llm.google_api_key = effective_key
            if hasattr(rag_agent, "embeddings"):
                rag_agent.embeddings.google_api_key = effective_key

        # Route 1: ReAct Agent Chat
        if parsed.path == "/api/chat":
            user_msg = req_json.get("message", "").strip()
            if not user_msg:
                self._send_json({"error": "Empty message"}, status=HTTPStatus.BAD_REQUEST)
                return

            print(f"\n[API /api/chat] Received prompt: {user_msg}")
            try:
                raw_ans = react_agent.run_turn(user_msg)
                
                # Clean and parse JSON response
                cleaned = raw_ans.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                try:
                    ans_data = json.loads(cleaned)
                except Exception:
                    ans_data = {
                        "query": user_msg,
                        "final_answer": raw_ans,
                        "tools_used": []
                    }

                self._send_json(ans_data)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Route 2: Student RAG Search
        if parsed.path == "/api/rag":
            query = req_json.get("query", "").strip()
            if not query:
                self._send_json({"error": "Empty query"}, status=HTTPStatus.BAD_REQUEST)
                return

            print(f"\n[API /api/rag] Received RAG query: {query}")
            try:
                rag_result = rag_agent.answer_query(query)
                self._send_json(rag_result)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Route Not Found
        self._send_json({"error": "Endpoint not found"}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[Server] {self.address_string()} - {format % args}\n")


def start_server():
    server_address = (HOST, PORT)
    try:
        httpd = ThreadingHTTPServer(server_address, AgentHTTPHandler)
    except OSError:
        # If port 8000 is busy, try port 8080
        fallback_port = 8080
        httpd = ThreadingHTTPServer((HOST, fallback_port), AgentHTTPHandler)
        print(f"[Notice] Port {PORT} was busy. Using fallback port {fallback_port}.")

    actual_port = httpd.server_port
    url = f"http://localhost:{actual_port}"

    print("=" * 70)
    print("  [SUCCESS] AGENTIC AI & RAG WEB STUDIO IS RUNNING!")
    print(f"  [URL] Open in your browser: {url}")
    print("  Zero Extra Frameworks | Pure Python Standard Library")
    print("=" * 70)
    print("Press Ctrl+C in terminal to stop server.\n")

    # Optionally auto-open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
