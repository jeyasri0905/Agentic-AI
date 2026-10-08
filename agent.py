"""
agent.py - Framework-Free Agentic AI with Gemini & Arithmetic Calculator
Author: Agentic AI Systems Architect

A pure Python Agentic AI implementation demonstrating:
1. Native Tool / Function Calling orchestration loop
2. Multi-turn conversation history management
3. Safe execution of arithmetic tools
4. Zero third-party dependencies (no LangChain, no LlamaIndex)
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from calculator import CALCULATOR_TOOL_DECLARATION, calculate
from datetime_tool import DATETIME_TOOL_DECLARATION, get_current_datetime
from document_search import DOCUMENT_SEARCH_TOOL_DECLARATION, document_search
from gemini_client import GeminiAPIError, GeminiClient
from web_search import WEB_SEARCH_TOOL_DECLARATION, web_search

SYSTEM_PROMPT = """You are an autonomous AI Agent built on the ReAct (Reasoning + Acting) architecture.
You are equipped with four specialized tools:
1. 'calculate': Performs exact mathematical calculations, arithmetic, formulas, roots, and percentages.
2. 'web_search': Searches the web for facts, definitions, current data, live information, or real-world numbers.
3. 'get_current_datetime': Gets the current live date, time, day of the week, and timezone.
4. 'document_search': Searches local files, policies, reports, and notes in the documents directory.

The ReAct Paradigm:
For any task, reason and act iteratively:
- THOUGHT (Reasoning): Analyze the user's request. Determine what is known, what information is missing, and why a specific tool is required.
- ACTION: Call the selected tool with precise parameters.
- OBSERVATION: Inspect the returned tool results and reason about the next step.
- REPEAT: If a multi-step task requires further actions, repeat Thought -> Action -> Observation.

Output Format:
You MUST ALWAYS format your FINAL response strictly as a JSON object with this structure:
{
  "query": "<user input>",
  "react_steps": [
    {
      "step": 1,
      "thought": "<reasoning for why this action was chosen>",
      "action": "<tool called and parameters>",
      "observation": "<summary of what was observed from the tool>"
    }
  ],
  "tools_used": ["<list of tools called, e.g. 'calculate', 'web_search', 'get_current_datetime', 'document_search', or empty>"],
  "expression": "<evaluated expression, or null>",
  "result": <numerical result, or null>,
  "final_answer": "<clear, comprehensive final response to the user>"
}
Return only valid JSON. Do not include markdown code block formatting or conversational text outside the JSON.
"""


class CalculatorAgent:
    """
    Autonomous agent coordinating Google AI Studio Gemini with deterministic tools.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        system_instruction: str = SYSTEM_PROMPT,
        verbose: bool = True
    ):
        self.client = GeminiClient(api_key=api_key, model=model)
        self.system_instruction = system_instruction
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []

        # Tool registry: function declaration schemas & local execution callables
        self.tools_schema = [
            {"functionDeclarations": [
                CALCULATOR_TOOL_DECLARATION,
                WEB_SEARCH_TOOL_DECLARATION,
                DATETIME_TOOL_DECLARATION,
                DOCUMENT_SEARCH_TOOL_DECLARATION
            ]}
        ]
        self.tool_dispatch: Dict[str, Callable] = {
            "calculate": calculate,
            "web_search": web_search,
            "get_current_datetime": get_current_datetime,
            "document_search": document_search
        }

    def reset_conversation(self) -> None:
        """Clears multi-turn conversation memory."""
        self.history = []

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run_turn(self, user_message: str, max_steps: int = 8) -> str:
        """
        Executes an agentic reasoning and action turn.
        
        Args:
            user_message: The prompt or query from the user.
            max_steps: Guardrail against infinite tool execution loops.

        Returns:
            The agent's final text response.
        """
        # 1. Append user input to multi-turn conversation history
        self.history.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        step_count = 0
        while step_count < max_steps:
            step_count += 1

            # 2. Query Gemini LLM with full context and tool declarations
            response = self.client.generate_content(
                contents=self.history,
                tools=self.tools_schema,
                system_instruction=self.system_instruction,
                temperature=0.1
            )

            candidates = response.get("candidates", [])
            if not candidates:
                raise GeminiAPIError("No response candidates received from Gemini API.")

            model_content = candidates[0].get("content", {})
            parts = model_content.get("parts", [])

            # Append the model's message (which could be text, tool calls, or both) to history
            self.history.append({
                "role": "model",
                "parts": parts
            })

            # 3. Detect and collect function call requests
            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if not function_calls:
                # Terminal step: The model concluded reasoning and provided a final text answer
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                return "\n".join(text_parts).strip()

            # Detect model's reasoning/thoughts if emitted
            reasoning_texts = [p.get("text", "").strip() for p in parts if "text" in p and p.get("text", "").strip()]

            # 4. Execute tool calls locally following ReAct (Reason + Action + Observation)
            response_parts = []
            for call in function_calls:
                func_name = call.get("name")
                func_args = call.get("args", {})

                self._log(f"\n--- [ReAct Step {step_count}] ---")
                if reasoning_texts:
                    self._log(f"[THOUGHT / REASON] : {' '.join(reasoning_texts)}")
                else:
                    self._log(f"[THOUGHT / REASON] : Need to execute '{func_name}' to make progress on the request.")
                
                self._log(f"[ACTION]           : Calling tool {func_name}({func_args})")

                if func_name in self.tool_dispatch:
                    tool_fn = self.tool_dispatch[func_name]
                    try:
                        tool_result = tool_fn(**func_args)
                    except Exception as err:
                        tool_result = {"status": "error", "error": str(err)}
                else:
                    tool_result = {"status": "error", "error": f"Unknown tool: '{func_name}'"}

                self._log(f"[OBSERVATION]      : {tool_result}")

                # Format response part according to Gemini Function Calling schema
                response_parts.append({
                    "functionResponse": {
                        "name": func_name,
                        "response": tool_result
                    }
                })

            # 5. Feed tool output back to Gemini as role 'user'
            self.history.append({
                "role": "user",
                "parts": response_parts
            })

        return "Agent halted: Maximum tool iteration depth reached without a final response."


def interactive_cli():
    """Interactive Command Line Interface for chatting with the Agent."""
    print("=" * 70)
    print("  GEMINI ReAct AGENT (Reasoning + Acting)")
    print("  Zero Frameworks | Pure Python | Tools: Calculator + Web + Date")
    print("=" * 70)
    print("Commands:")
    print("  /clear   - Reset conversation memory")
    print("  /history - Show number of turns in memory")
    print("  /model   - Display current model")
    print("  /exit    - Terminate session")
    print("-" * 70)

        # API Key configuration
    api_key = "AQ.Ab8RN6JYtoXDU0h1BYtDraFHAv1b6v69VIb3PMntbf5j0kp5BA"

    if not api_key:
        print("\n[!] No GEMINI_API_KEY detected.")
        api_key = input("Please enter your Google AI Studio API key: ").strip()
        if not api_key:
            print("Error: API key is required to proceed. Exiting.")
            sys.exit(1)
    

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

    try:
        agent = CalculatorAgent(api_key=api_key, model=model_name, verbose=True)
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    print(f"\n[+] Agent initialized successfully with model: {model_name}")
    print("Tools available: 'calculate', 'web_search', 'get_current_datetime', 'document_search'")
    print("Examples:")
    print("  - Math:     'What is (45.5 * 12) + sqrt(144)?'")
    print("  - Web:      'Who is the CEO of Google?'")
    print("  - Date:     'What is today's date and current time?'")
    print("  - Doc:      'What is our company annual vacation leave policy?'")
    print("  - Multi:    'Look up Q3 profit from financial documents and calculate 20% of it'\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Exiting session. Goodbye!")
            break
        elif cmd == "/clear":
            agent.reset_conversation()
            print("[+] Conversation memory cleared.")
            continue
        elif cmd == "/history":
            print(f"[i] Conversation turns in memory: {len(agent.history)}")
            continue
        elif cmd == "/model":
            print(f"[i] Active model: {agent.client.model}")
            continue

        try:
            answer = agent.run_turn(user_input)
            cleaned = answer.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            try:
                parsed = json.loads(cleaned)
                output_str = json.dumps(parsed, indent=2)
            except Exception:
                output_str = answer

            print(f"\nAgent >\n{output_str}")
        except GeminiAPIError as api_err:
            print(f"\n[API Error] {api_err}")
        except Exception as ex:
            print(f"\n[Unexpected Error] {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Calculator Agent (No Frameworks)")
    parser.add_argument("--prompt", type=str, help="Single-shot prompt to execute")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash", help="Gemini model name")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose tool logs")
    args = parser.parse_args()

    if args.prompt:
        # Single-shot execution mode
        agent = CalculatorAgent(model=args.model, verbose=not args.quiet)
        try:
            result = agent.run_turn(args.prompt)
            print("\nFinal Answer:\n" + result)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        interactive_cli()
