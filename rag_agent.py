"""
rag_agent.py - Retrieval-Augmented Generation (RAG) for Student Database
Implements full production RAG pipeline:
1. Document Ingestion & Chunking (Dummy Student Database)
2. Dense Vector Embeddings using Google Gemini (models/gemini-embedding-001) + Hybrid Keyword Search
3. Top-K Vector Retrieval & Context Augmentation
4. LLM Generation via Gemini with Structured JSON Output
"""

import json
import os
import re
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# ==============================================================================
# 1. API KEY CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = "AQ.Ab8RN6JYtoXDU0h1BYtDraFHAv1b6v69VIb3PMntbf5j0kp5BA" or os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = "models/gemini-embedding-001"

if not GEMINI_API_KEY:
    print("\n[!] No GEMINI_API_KEY detected.")
    GEMINI_API_KEY = input("Please enter your Google AI Studio API key: ").strip()
    if not GEMINI_API_KEY:
        print("Error: API key is required. Exiting.")
        sys.exit(1)

DOC_TXT_PATH = Path(__file__).parent / "documents" / "student_database.txt"
DOC_CSV_PATH = Path(__file__).parent / "documents" / "student_database.csv"


# ==============================================================================
# 2. DOCUMENT INGESTION & CHUNKING
# ==============================================================================
class DocumentStore:
    """Stores student records as individual chunks for retrieval."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._load_documents()

    def _load_documents(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Database document not found at {self.file_path}")

        content = self.file_path.read_text(encoding="utf-8")
        
        # Split into individual student records based on 'Student Record STU...'
        raw_records = re.split(r"(?=Student Record STU\d+:)", content)
        
        for raw in raw_records:
            cleaned = raw.strip()
            if not cleaned or "Student Record" not in cleaned:
                continue

            # Extract student ID and name for metadata
            stu_id_match = re.search(r"Student ID:\s*(STU\d+)", cleaned)
            name_match = re.search(r"Name:\s*([^\n]+)", cleaned)
            dept_match = re.search(r"Department:\s*([^\n]+)", cleaned)

            stu_id = stu_id_match.group(1) if stu_id_match else "UNKNOWN"
            name = name_match.group(1).strip() if name_match else "UNKNOWN"
            dept = dept_match.group(1).strip() if dept_match else "UNKNOWN"

            self.chunks.append(cleaned)
            self.metadata.append({
                "student_id": stu_id,
                "name": name,
                "department": dept
            })

        print(f"[Document Store] Loaded {len(self.chunks)} student document chunks.")


# ==============================================================================
# 3. VECTOR EMBEDDINGS & RETRIEVER
# ==============================================================================
class StudentRAGRetriever:
    """Hybrid Retriever combining dense vector cosine similarity with keyword search."""

    def __init__(self, doc_store: DocumentStore, embedding_client: GoogleGenerativeAIEmbeddings):
        self.store = doc_store
        self.embedding_client = embedding_client
        self.chunk_embeddings: List[np.ndarray] = []
        self._build_vector_index()

    def _build_vector_index(self):
        print(f"[RAG Indexer] Generating dense vector embeddings via '{EMBEDDING_MODEL}'...")
        try:
            vectors = self.embedding_client.embed_documents(self.store.chunks)
            self.chunk_embeddings = [np.array(v, dtype=np.float32) for v in vectors]
            print(f"[RAG Indexer] Index complete! Stored {len(self.chunk_embeddings)} vectors (dim: {len(vectors[0])}).")
        except Exception as e:
            print(f"[Warning] Dense embedding generation failed ({e}). Falling back to lexical hybrid retriever.")
            self.chunk_embeddings = []

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Retrieves top-K most relevant student chunks for a query.
        Returns list of (chunk_text, metadata, score).
        """
        query_clean = query.lower()
        scores = np.zeros(len(self.store.chunks), dtype=np.float32)

        # 1. Dense Semantic Similarity
        if self.chunk_embeddings:
            try:
                q_vec = np.array(self.embedding_client.embed_query(query), dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    for i, doc_vec in enumerate(self.chunk_embeddings):
                        d_norm = np.linalg.norm(doc_vec)
                        if d_norm > 0:
                            cosine_sim = float(np.dot(q_vec, doc_vec) / (q_norm * d_norm))
                            scores[i] += cosine_sim * 0.7  # 70% weight for semantic similarity
            except Exception:
                pass

        # 2. Keyword & Entity Booster (30% weight)
        for i, meta in enumerate(self.store.metadata):
            name_lower = meta["name"].lower()
            id_lower = meta["student_id"].lower()
            dept_lower = meta["department"].lower()

            if name_lower in query_clean:
                scores[i] += 0.8
            if id_lower in query_clean:
                scores[i] += 0.9
            if dept_lower in query_clean:
                scores[i] += 0.4

        # For aggregate / whole dataset queries (e.g. "highest", "how many", "all students")
        aggregate_keywords = ["highest", "lowest", "how many", "all", "list", "top", "below", "above"]
        if any(w in query_clean for w in aggregate_keywords):
            top_k = min(len(self.store.chunks), 25)  # include full table for aggregates

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                self.store.chunks[idx],
                self.store.metadata[idx],
                float(scores[idx])
            ))
        return results


# ==============================================================================
# 4. RAG GENERATION ENGINE (LLM + CONTEXT AUGMENTATION)
# ==============================================================================
class StudentRAGAgent:
    """End-to-End RAG Agent retrieving student data and returning structured JSON."""

    def __init__(self):
        self.doc_store = DocumentStore(DOC_TXT_PATH)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )
        self.retriever = StudentRAGRetriever(self.doc_store, self.embeddings)
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0.0
        )

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the RAG cycle:
        1. Retrieve top context chunks
        2. Augment prompt with retrieved documents
        3. Generate response using Gemini
        4. Return strictly structured JSON
        """
        retrieved_items = self.retriever.retrieve(user_query, top_k=4)
        
        # Build augmented context
        context_blocks = [chunk for chunk, meta, score in retrieved_items]
        augmented_context = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a strict, factual RAG Question-Answering Agent for a Student Academic Database. "
            "Answer the user query ONLY using the provided retrieved context. "
            "Do NOT make up facts. If information is not in the context, state it. "
            "You MUST format your output strictly as a JSON object with this schema:\n"
            "{\n"
            '  "query": "<user question>",\n'
            '  "answer": "<direct, clear, concise answer>",\n'
            '  "details": {\n'
            '    "student_name": "<name or null>",\n'
            '    "student_id": "<ID or null>",\n'
            '    "relevant_data": "<specific values retrieved>"\n'
            '  },\n'
            '  "confidence": "high"\n'
            "}\n"
            "Return only valid JSON without markdown formatting."
        )

        prompt = f"RETRIEVED STUDENT DATABASE CONTEXT:\n{augmented_context}\n\nUSER QUESTION: {user_query}"

        response = self.llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt)
        ])

        # Clean JSON output
        raw_text = response.content
        if isinstance(raw_text, list):
            raw_text = " ".join([item.get("text", "") for item in raw_text if isinstance(item, dict) and "text" in item])
        
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed_json = json.loads(cleaned)
        except Exception:
            parsed_json = {
                "query": user_query,
                "answer": cleaned,
                "confidence": "high"
            }

        # Attach retrieval metadata
        parsed_json["retrieved_sources"] = [
            {"student_id": meta["student_id"], "name": meta["name"]}
            for chunk, meta, score in retrieved_items[:3]
        ]

        return parsed_json


# ==============================================================================
# 5. INTERACTIVE CLI WITH SUGGESTED RAG QUESTIONS
# ==============================================================================
SUGGESTED_QUESTIONS = [
    "1. What is Priya Sharma's attendance and overall mark?",
    "2. Who has the highest overall mark?",
    "3. Which students belong to the CSE department?",
    "4. List students whose attendance is below 75%.",
    "5. Who scored above 90 in AI?",
    "6. How many students are placement eligible?",
    "7. What is Arjun Kumar's DSA mark?",
    "8. Which student has the highest attendance?",
    "9. Find all AIDS students with an overall mark above 80.",
    "10. What is the phone number of Fathima N?"
]


def interactive_cli():
    print("=" * 75)
    print("  STUDENT DATABASE RAG AGENT (LangChain + Gemini Embeddings)")
    print("  Retrieval-Augmented Generation over Student Dataset")
    print("=" * 75)
    print("Document Loaded: Dummy Student Database (25 Students STU001 - STU025)")
    print("\nSuggested Questions from the Document:")
    for sq in SUGGESTED_QUESTIONS:
        print(f"  {sq}")
    print("\nType any number 1-10 to run a suggested question, or type your own question.")
    print("Type 'exit' to quit.\n" + "-" * 75)

    agent = StudentRAGAgent()

    while True:
        try:
            user_input = input("\nRAG Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "/exit"):
            print("Exiting. Goodbye!")
            break

        # Check if user typed a number 1-10
        if user_input.isdigit() and 1 <= int(user_input) <= 10:
            user_input = SUGGESTED_QUESTIONS[int(user_input) - 1].split(". ", 1)[1]
            print(f"[Selected Question]: {user_input}")

        print("\n[RAG Pipeline] Retrieving relevant chunks & generating answer...")
        result = agent.answer_query(user_input)
        print("\n--- [RAG JSON RESULT] ---")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    interactive_cli()
