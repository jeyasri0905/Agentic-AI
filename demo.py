"""
demo.py - Programmatic Usage Example
Demonstrates how to embed CalculatorAgent into your own Python applications.
"""

import os
from agent import CalculatorAgent

def main():
    # Retrieve API key from environment variable or prompt
    api_key ="AQ.Ab8RN6JYtoXDU0h1BYtDraFHAv1b6v69VIb3PMntbf5j0kp5BA"
    if not api_key:
        api_key = input("Enter your Google AI Studio API key: ").strip()
        if not api_key:
            print("Error: GEMINI_API_KEY is required to run the demo.")
            return

    # Initialize the Agent
    agent = CalculatorAgent(api_key=api_key, model="gemini-3.8-flash", verbose=True)

    test_queries = [
        "Calculate the compound expression: (145 * 28) - (350 / 7) + 2^5",
        "If a store has a 35% discount on an item costing $240 and sales tax is 8%, what is the final price?",
        "What is the square root of 576 added to 15 factorial divided by 14 factorial?"
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"User Query: {query}")
        print("=" * 60)
        answer = agent.run_turn(query)
        print(f"\nFinal Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
