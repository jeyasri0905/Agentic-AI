"""
two_input_agent.py - Dedicated 2-Input Arithmetic Agent
Author: Senior Agentic AI Architect

Collects 2 user inputs and an arithmetic operation, routes through
Google AI Studio Gemini, and executes the deterministic calculation tool.
"""

import os
import sys
from agent import CalculatorAgent, GeminiAPIError


def run_two_input_agent():
    print("=" * 70)
    print("     TWO-INPUT ARITHMETIC AGENT (Framework-Free Pure Python)")
    print("     Powered by Google AI Studio Gemini & Safe AST Calculator")
    print("=" * 70)

    # 1. Resolve API Key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        # Check if set in agent.py line 160 (fallback)
        try:
            from agent import api_key as default_key
            if default_key and default_key.startswith("AIzaSy"):
                api_key = default_key
        except Exception:
            pass

    if not api_key or not api_key.startswith("AIzaSy"):
        print("\n[!] Please provide your Google AI Studio Gemini API Key.")
        print("    (It must start with 'AIzaSy' from https://aistudio.google.com/app/apikey)")
        api_key = input("Enter API Key: ").strip().strip("'\"")
        if not api_key:
            print("Error: API Key is required to proceed.")
            sys.exit(1)

    # 2. Initialize Agent with gemini-3.8-flash
    try:
        agent = CalculatorAgent(api_key=api_key, model="gemini-3.8-flash", verbose=True)
        print("\n[+] Agent connected successfully! Model: gemini-3.8-flash")
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    print("\nYou can perform operations on any 2 numbers or expressions.")
    print("Type 'exit' or 'q' anytime to quit.\n")

    # 3. Interactive 2-Input Loop
    while True:
        try:
            print("-" * 50)
            val1 = input("Enter Input 1 (e.g. 50, 12.5, sqrt(144)): ").strip()
            if val1.lower() in ("exit", "quit", "q"):
                print("Exiting session. Goodbye!")
                break
            if not val1:
                print("Input 1 cannot be empty.")
                continue

            val2 = input("Enter Input 2 (e.g. 25, 4, 3^2): ").strip()
            if val2.lower() in ("exit", "quit", "q"):
                print("Exiting session. Goodbye!")
                break
            if not val2:
                print("Input 2 cannot be empty.")
                continue

            print("\nAvailable Operations: + (add), - (subtract), * (multiply), / (divide),")
            print("                     // (floor div), % (modulo), ** or ^ (power), average, percentage")
            op = input("Enter Operation: ").strip()
            if op.lower() in ("exit", "quit", "q"):
                print("Exiting session. Goodbye!")
                break
            if not op:
                op = "+"

            # 4. Formulate the prompt for the Agent
            prompt = (
                f"You have received two inputs from the user:\n"
                f"- Input 1: {val1}\n"
                f"- Input 2: {val2}\n"
                f"The requested arithmetic operation is: '{op}'.\n"
                f"Instructions:\n"
                f"1. Formulate the exact mathematical expression.\n"
                f"2. You MUST use the 'calculate' tool to evaluate it.\n"
                f"3. Provide the final calculated result and explanation in clear, simple English."
            )

            print(f"\n[Sending to Agent] Performing '{op}' on ({val1}) and ({val2})...\n")

            # 5. Agent Reason -> Action -> Observation -> Final Answer Loop
            answer = agent.run_turn(prompt)
            print("\n" + "=" * 50)
            print("RESULT:")
            print(answer)
            print("=" * 50)

        except GeminiAPIError as err:
            print(f"\n[Gemini API Error] {err}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session. Goodbye!")
            break
        except Exception as ex:
            print(f"\n[Unexpected Error] {ex}")


if __name__ == "__main__":
    run_two_input_agent()
