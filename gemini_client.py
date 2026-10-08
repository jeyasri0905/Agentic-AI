"""
gemini_client.py - Zero-Dependency Google AI Studio Gemini API Client
Author: Agentic AI Systems Architect

Communicates directly with Google AI Studio's Gemini REST endpoints via Python's
standard library `urllib.request`. Zero external dependencies or bloated frameworks.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class GeminiAPIError(Exception):
    """Raised when the Gemini REST API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class GeminiClient:
    """
    Direct REST API client for Google AI Studio Gemini models.
    Supports system instructions, multi-turn dialogue, and native function calling.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        timeout: int = 30
    ):
        """
        Initialize the Gemini REST client.
        
        Args:
            api_key: Google AI Studio API key. If omitted, reads from GEMINI_API_KEY env var.
            model: Gemini model identifier (e.g., 'gemini-3.8-flash').
            timeout: Network request timeout in seconds.
        """
        raw_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip().strip("'\"")
        if not self.api_key:
            raise ValueError(
                "Gemini API Key is required. Please pass it to GeminiClient "
                "or set the 'GEMINI_API_KEY' environment variable."
            )
        self.model = model
        self.timeout = timeout

    def generate_content(
        self,
        contents: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Sends a request to the generateContent endpoint.
        
        Args:
            contents: Multi-turn message history following the Gemini REST spec.
            tools: List of tool declarations (e.g. functionDeclarations).
            system_instruction: System prompt guiding the agent's behavior.
            temperature: Sampling temperature (0.0 to 2.0).

        Returns:
            Parsed JSON response from the Gemini API.
        """
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        if tools:
            payload["tools"] = tools

        json_data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url=url,
            data=json_data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
                "User-Agent": "AgenticAI-CalculatorAgent/1.0"
            },
            method="POST"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                    return json.loads(response_body)
            except urllib.error.HTTPError as http_err:
                error_body = http_err.read().decode("utf-8")
                try:
                    error_json = json.loads(error_body)
                    err_msg = error_json.get("error", {}).get("message", error_body)
                    err_status = error_json.get("error", {}).get("status", str(http_err.code))
                except Exception:
                    err_msg = error_body
                    err_status = str(http_err.code)

                # Transient high-demand or rate-limit spike (503 / 429 / 500)
                if http_err.code in (503, 429, 500) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"\n[Temporary Google AI Studio Load Spike] Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(wait_time)
                    continue

                raise GeminiAPIError(
                    f"Gemini API request failed [{err_status}]: {err_msg}",
                    status_code=http_err.code,
                    details={"raw_error": error_body}
                ) from http_err
            except urllib.error.URLError as url_err:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                raise GeminiAPIError(f"Network connection failed: {url_err.reason}") from url_err
