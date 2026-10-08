"""
Gemini Calculator Agent
========================
A zero-framework Agentic AI implementation using Google's official 'google-genai' SDK.
Demonstrates:
1. Tool declaration formatted in JSON schema.
2. LLM tool-calling (function calling) where the LLM decides the tool arguments.
3. Local execution of the tool with inputs received from the LLM.
4. Sending tool output back to the LLM to synthesize the final answer.
"""

import json
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types

# ==============================================================================
# 1. API KEY CONFIGURATION
# Add your Gemini API key from AI Studio below, or set GEMINI_API_KEY env variable.
# ==============================================================================
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"

# If the placeholder is unchanged, try reading from environment variable
if GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        GEMINI_API_KEY = env_key

# Model to use (gemini-3.5-flash-lite has higher free-tier quota)
MODEL_NAME = "gemini-3.5-flash-lite"

# ==============================================================================
# 2. CALCULATOR TOOL DEFINITION IN JSON FORMAT
# ==============================================================================
# This is the exact JSON schema that the LLM uses to understand:
# - What the tool does (description)
# - What arguments to generate (operation, a, b)
# ==============================================================================
TOOL_JSON_PATH = Path(__file__).parent / "calculator_tool.json"

if TOOL_JSON_PATH.exists():
    with open(TOOL_JSON_PATH, "r", encoding="utf-8") as f:
        CALCULATOR_TOOL_SCHEMA = json.load(f)
else:
    # Fallback embedded JSON if file is not found
    CALCULATOR_TOOL_SCHEMA = {
        "name": "calculator",
        "description": "Performs basic arithmetic operations between two numbers: addition, subtraction, multiplication, division, and exponentiation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "operation": {
                    "type": "STRING",
                    "description": "The arithmetic operation to perform.",
                    "enum": ["add", "subtract", "multiply", "divide", "power"]
                },
                "a": {
                    "type": "NUMBER",
                    "description": "The first number (left operand)."
                },
                "b": {
                    "type": "NUMBER",
                    "description": "The second number (right operand)."
                }
            },
            "required": ["operation", "a", "b"]
        }
    }


# ==============================================================================
# 3. LOCAL TOOL IMPLEMENTATION
# This function is executed locally when the LLM requests a calculator call.
# ==============================================================================
def execute_calculator(operation: str, a: float, b: float) -> dict:
    """Executes the calculator operation requested by the LLM."""
    try:
        a = float(a)
        b = float(b)
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Division by zero is undefined."}
            result = a / b
        elif operation == "power":
            result = a ** b
        else:
            return {"error": f"Unsupported operation '{operation}'."}
            
        # Format integer results neatly (e.g., 42 instead of 42.0)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
            
        return {"result": result}
    except Exception as exc:
        return {"error": str(exc)}


# ==============================================================================
# 4. AGENT LOGIC (AGENTIC FUNCTION CALLING LOOP)
# ==============================================================================
def create_agent_chat(client: genai.Client):
    """Creates a chat session with the calculator tool attached."""
    # Convert JSON schema definition into a GenAI Tool
    calc_declaration = types.FunctionDeclaration(
        name=CALCULATOR_TOOL_SCHEMA["name"],
        description=CALCULATOR_TOOL_SCHEMA["description"],
        parameters=CALCULATOR_TOOL_SCHEMA["parameters"]
    )
    tools = [types.Tool(function_declarations=[calc_declaration])]
    
    # System instruction giving the agent its identity and guidelines
    system_instruction = (
        "You are an accurate math assistant agent. "
        "Whenever a user asks for a math computation, you MUST use the 'calculator' tool "
        "to perform the calculation instead of computing it in your head. "
        "For compound operations (e.g. (15 + 5) * 3), call the calculator tool step by step."
    )

    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=system_instruction,
        temperature=0.0
    )
    
    return client.chats.create(model=MODEL_NAME, config=config)


def run_agentic_turn(chat, user_input: str) -> str:
    """
    Executes one turn of the agent:
    1. Sends user message to LLM.
    2. While the LLM decides to invoke tools:
       a. Intercepts the tool call arguments generated by LLM.
       b. Executes local Python tool.
       c. Sends tool response back to LLM.
    3. Returns the final text response from the LLM.
    """
    print(f"\n[USER] {user_input}")
    print("[AGENT] Sending request to Gemini...")
    
    response = chat.send_message(user_input)
    
    # Loop as long as the LLM requests function calls (handles multi-step reasoning)
    step_count = 1
    while response.function_calls:
        tool_responses = []
        for call in response.function_calls:
            print(f"\n--- [STEP {step_count}: LLM DECISION] ---")
            print(f"Tool Requested : {call.name}")
            print(f"Arguments (JSON): {json.dumps(call.args, indent=2)}")
            
            if call.name == "calculator":
                op = call.args.get("operation")
                a = call.args.get("a")
                b = call.args.get("b")
                
                print(f"[EXECUTING TOOL] Running local calculator with ({a} {op} {b})...")
                calc_result = execute_calculator(op, a, b)
                print(f"[TOOL OUTPUT]   {json.dumps(calc_result)}")
                
                # Format the response back to Gemini
                part = types.Part.from_function_response(
                    name=call.name,
                    response=calc_result
                )
                tool_responses.append(part)
            else:
                # Unknown tool fallback
                unknown_res = {"error": f"Tool '{call.name}' is not recognized."}
                part = types.Part.from_function_response(
                    name=call.name,
                    response=unknown_res
                )
                tool_responses.append(part)
        
        step_count += 1
        print("[AGENT] Feeding tool results back to LLM...")
        response = chat.send_message(tool_responses)
        
    final_text = response.text or "Done."
    print("\n--- [FINAL ANSWER FROM AGENT] ---")
    print(final_text)
    return final_text


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
def main():
    print("=" * 60)
    print("      GEMINI AGENTIC AI - BASIC CALCULATOR AGENT      ")
    print("=" * 60)
    
    # Check API Key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE":
        print("\n[ERROR] Missing Gemini API key!")
        print("Please open app.py and set GEMINI_API_KEY = \"your_api_key_here\"")
        print("Or set the environment variable: set GEMINI_API_KEY=\"your_key\"")
        print("\nExiting...")
        sys.exit(1)
        
    # Initialize Google GenAI client (pure SDK, zero extra agent frameworks)
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Gemini Client: {e}")
        sys.exit(1)

    print("[SUCCESS] Connected to Gemini API.")
    print("Calculator Tool loaded from JSON schema.")
    print("Type your questions in natural language. Type 'exit' or 'quit' to end.\n")
    print("Examples:")
    print("  - What is 45 multiplied by 89?")
    print("  - What is 2 raised to the power of 12?")
    print("  - Calculate (125 + 75) divided by 5")
    print("  - If I have 1500 and spend 325, how much is left?")
    print("-" * 60)
    
    chat = create_agent_chat(client)
    
    while True:
        try:
            user_query = input("\nYou: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            run_agentic_turn(chat, user_query)
            
        except KeyboardInterrupt:
            print("\nExiting session...")
            break
        except Exception as err:
            print(f"\n[ERROR occurred during execution]: {err}")


if __name__ == "__main__":
    main()
