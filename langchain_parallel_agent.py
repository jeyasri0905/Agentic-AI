"""
langchain_parallel_agent.py - Multi-Task Parallel Execution using LangChain
Demonstrates running multiple agentic tasks simultaneously using LangChain:
1. Parallel Batching with LangChain's Agent `.batch()`
2. Concurrent sub-task execution with `RunnableParallel`
3. Returns comprehensive structured JSON output.
"""

import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

# LangChain Imports
from langchain_core.tools import tool
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Import our existing deterministic tools
from calculator import calculate
from datetime_tool import get_current_datetime
from document_search import document_search
from web_search import web_search

# ==============================================================================
# 1. API KEY CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = "AQ.Ab8RN6JYtoXDU0h1BYtDraFHAv1b6v69VIb3PMntbf5j0kp5BA" or os.environ.get("GEMINI_API_KEY")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not GEMINI_API_KEY:
    print("\n[!] No GEMINI_API_KEY detected.")
    GEMINI_API_KEY = input("Please enter your Google AI Studio API key: ").strip()
    if not GEMINI_API_KEY:
        print("Error: API key is required. Exiting.")
        sys.exit(1)


# ==============================================================================
# 2. LANGCHAIN TOOL DEFINITIONS
# ==============================================================================
@tool
def calculate_tool(expression: str) -> str:
    """Evaluates mathematical expressions (e.g. '45 * 12', 'sqrt(144) + 10', '2^8')."""
    res = calculate(expression)
    return json.dumps(res)


@tool
def web_search_tool(query: str) -> str:
    """Searches the web for facts, encyclopedic information, definitions, or current events."""
    res = web_search(query)
    return json.dumps(res)


@tool
def datetime_tool(timezone: str = "local") -> str:
    """Returns the current date, time, day of week, month, and year."""
    res = get_current_datetime(timezone=timezone)
    return json.dumps(res)


@tool
def document_search_tool(query: str, filename_filter: str = "") -> str:
    """Searches local company documents, policies, reports, and notes in the documents directory."""
    res = document_search(query=query, filename_filter=filename_filter or None)
    return json.dumps(res)


TOOLS = [calculate_tool, web_search_tool, datetime_tool, document_search_tool]


# ==============================================================================
# 3. INITIALIZE LANGCHAIN LLM & RE-ACT AGENT
# ==============================================================================
def create_langchain_agent():
    """Initializes the ChatGoogleGenerativeAI LLM and LangGraph ReAct agent."""
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )
    # create_react_agent attaches the tools and wires the reasoning/action graph
    agent = create_react_agent(llm, TOOLS)
    return agent


# ==============================================================================
# 4. PARALLEL MULTI-TASK EXECUTION LOGIC
# ==============================================================================
def run_tasks_in_parallel(agent, task_queries: List[str]) -> Dict[str, Any]:
    """
    Executes multiple distinct tasks concurrently at the same time using LangChain's `.batch()`.
    """
    start_time = time.time()
    
    # Prepare batch inputs for LangChain
    batch_inputs = [
        {"messages": [{"role": "user", "content": q}]}
        for q in task_queries
    ]

    print(f"\n[LangChain Engine] Launching {len(task_queries)} tasks in PARALLEL...")
    for idx, q in enumerate(task_queries, 1):
        print(f"  -> Task {idx}: '{q}'")
    
    # LangChain's .batch() runs all items concurrently across worker threads
    batch_results = agent.batch(batch_inputs)
    
    elapsed = round(time.time() - start_time, 2)
    print(f"[LangChain Engine] All {len(task_queries)} tasks finished in {elapsed}s!\n")

    # Format into structured JSON output
    formatted_tasks = []
    for idx, (query, res) in enumerate(zip(task_queries, batch_results), 1):
        messages = res.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # Extract clean text
        content = ""
        if last_message:
            if isinstance(last_message.content, list):
                text_items = [item.get("text", "") for item in last_message.content if isinstance(item, dict) and "text" in item]
                content = " ".join(text_items).strip()
            else:
                content = str(last_message.content).strip()

        # Count tool calls in this turn
        tools_called = []
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tools_called.append(tc.get("name"))

        formatted_tasks.append({
            "task_id": idx,
            "query": query,
            "tools_used": list(set(tools_called)),
            "answer": content
        })

    return {
        "status": "success",
        "execution_mode": "LangChain Parallel Batch",
        "total_tasks": len(task_queries),
        "total_execution_time_seconds": elapsed,
        "tasks": formatted_tasks
    }


# ==============================================================================
# 5. RUNNABLE PARALLEL DEMO (Single topic, concurrent branches)
# ==============================================================================
def run_runnable_parallel_pipeline(topic: str) -> Dict[str, Any]:
    """
    Demonstrates LangChain's RunnableParallel by running 3 branches simultaneously:
    1. A math estimation/calculation task
    2. A web search task
    3. A date-time stamp task
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )

    print(f"\n[RunnableParallel] Executing 3 parallel pipelines on topic: '{topic}'...")
    start_time = time.time()

    # Define parallel branches
    parallel_chain = RunnableParallel(
        summary=llm,
        math_check=(lambda x: f"Calculate the number of characters in '{x}' times 10: {len(x) * 10}"),
        timestamp=(lambda _: get_current_datetime())
    )

    result = parallel_chain.invoke(f"Provide a 1-sentence interesting scientific fact about {topic}.")
    elapsed = round(time.time() - start_time, 2)

    return {
        "status": "success",
        "execution_mode": "LangChain RunnableParallel",
        "topic": topic,
        "elapsed_seconds": elapsed,
        "parallel_outputs": {
            "summary": result["summary"].content if hasattr(result["summary"], "content") else str(result["summary"]),
            "math_check": result["math_check"],
            "timestamp": result["timestamp"]
        }
    }


# ==============================================================================
# 6. INTERACTIVE INTERFACE
# ==============================================================================
def interactive_cli():
    print("=" * 75)
    print("  LANGCHAIN MULTI-TASK PARALLEL AGENT")
    print("  Executes Multiple Tasks Simultaneously via LangChain .batch()")
    print("=" * 75)
    print("Model:", MODEL_NAME)
    print("Instructions:")
    print("  - To run MULTIPLE tasks at a time, separate them with a pipe '|' or semicolon ';'")
    print("    Example: 45 * 89 | Who founded Python? | What is today's date?")
    print("  - Type 'demo' to run a pre-configured 3-task parallel batch test.")
    print("  - Type 'pipeline <topic>' to test LangChain RunnableParallel.")
    print("  - Type 'exit' to quit.\n")

    agent = create_langchain_agent()

    while True:
        try:
            user_input = input("\nEnter Tasks > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "/exit"):
            print("Exiting. Goodbye!")
            break

        if cmd == "demo":
            demo_tasks = [
                "Calculate: 154 * 23",
                "Who founded Python programming language?",
                "What is the current date and time?"
            ]
            res = run_tasks_in_parallel(agent, demo_tasks)
            print("JSON Output:")
            print(json.dumps(res, indent=2))
            continue

        if cmd.startswith("pipeline"):
            parts = user_input.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else "Quantum Computing"
            res = run_runnable_parallel_pipeline(topic)
            print("JSON Output:")
            print(json.dumps(res, indent=2))
            continue

        # Split user input into multiple tasks if separated by | or ;
        if "|" in user_input:
            tasks = [t.strip() for t in user_input.split("|") if t.strip()]
        elif ";" in user_input:
            tasks = [t.strip() for t in user_input.split(";") if t.strip()]
        else:
            tasks = [user_input]

        res = run_tasks_in_parallel(agent, tasks)
        print("JSON Output:")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    interactive_cli()
