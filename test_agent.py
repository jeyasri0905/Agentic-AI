"""
test_agent.py - Comprehensive Unit and Security Tests for Agent & Calculator
Author: Agentic AI Systems Architect
"""

import math
import unittest
from unittest.mock import MagicMock, patch

from calculator import SecurityEvaluationError, calculate
from agent import CalculatorAgent


class TestSafeMathEvaluator(unittest.TestCase):
    """Verifies all arithmetic operations, math functions, and security bounds."""

    def test_basic_arithmetic(self):
        # Addition
        self.assertEqual(calculate("15 + 27")["result"], 42)
        # Subtraction
        self.assertEqual(calculate("100 - 45.5")["result"], 54.5)
        # Multiplication
        self.assertEqual(calculate("12 * 12")["result"], 144)
        # Division (float)
        self.assertEqual(calculate("100 / 4")["result"], 25)
        self.assertAlmostEqual(calculate("10 / 3")["result"], 3.3333333333333335)
        # Floor division
        self.assertEqual(calculate("10 // 3")["result"], 3)
        # Modulo
        self.assertEqual(calculate("17 % 5")["result"], 2)
        # Power via **
        self.assertEqual(calculate("2 ** 8")["result"], 256)
        # Power via ^
        self.assertEqual(calculate("3 ^ 4")["result"], 81)

    def test_order_of_operations_and_parentheses(self):
        self.assertEqual(calculate("2 + 3 * 4")["result"], 14)
        self.assertEqual(calculate("(2 + 3) * 4")["result"], 20)
        self.assertEqual(calculate("100 - 5 * (2 + 3) ** 2")["result"], -25)

    def test_unary_operations(self):
        self.assertEqual(calculate("-5 + 10")["result"], 5)
        self.assertEqual(calculate("+5 - -10")["result"], 15)
        self.assertEqual(calculate("-(3 + 2)")["result"], -5)

    def test_math_functions_and_constants(self):
        self.assertEqual(calculate("sqrt(144)")["result"], 12)
        self.assertEqual(calculate("factorial(5)")["result"], 120)
        self.assertEqual(calculate("abs(-42)")["result"], 42)
        self.assertEqual(calculate("round(3.14159, 2)")["result"], 3.14)
        self.assertEqual(calculate("log10(1000)")["result"], 3)
        self.assertAlmostEqual(calculate("sin(pi / 2)")["result"], 1.0)
        self.assertAlmostEqual(calculate("cos(0)")["result"], 1.0)
        self.assertAlmostEqual(calculate("exp(1)")["result"], math.e)

    def test_error_handling(self):
        # Division by zero
        res = calculate("5 / 0")
        self.assertEqual(res["status"], "error")
        self.assertIn("zero", res["error"].lower())

        # Syntax error
        res = calculate("2 + * 3")
        self.assertEqual(res["status"], "error")
        self.assertIn("syntax error", res["error"].lower())

    def test_security_sandboxing(self):
        """Ensure arbitrary code execution / system access is strictly blocked."""
        malicious_inputs = [
            "__import__('os').system('dir')",
            "open('test.txt', 'w')",
            "eval('2 + 2')",
            "exec('x = 10')",
            "[x for x in (1, 2, 3)]",
            "lambda x: x + 1",
            "import os",
            "__builtins__",
            "str.__class__",
            "(1).__class__.__bases__[0]",
        ]
        for payload in malicious_inputs:
            res = calculate(payload)
            self.assertEqual(res["status"], "error", f"Payload failed to be blocked: {payload}")


class TestAgentLoop(unittest.TestCase):
    """Tests the agentic tool orchestration loop using mock Gemini responses."""

    def test_agent_turn_with_tool_call(self):
        # Create agent with dummy key and verbose=False
        agent = CalculatorAgent(api_key="mock_key", verbose=False)

        # Mock responses:
        # Step 1: Model requests a tool call to 'calculate'
        # Step 2: Model finishes with final text answer
        mock_response_1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "calculate",
                                    "args": {"expression": "25 * 4 + 10"}
                                }
                            }
                        ],
                        "role": "model"
                    }
                }
            ]
        }

        mock_response_2 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "The result of 25 * 4 + 10 is 110."
                            }
                        ],
                        "role": "model"
                    }
                }
            ]
        }

        with patch.object(agent.client, "generate_content", side_effect=[mock_response_1, mock_response_2]):
            final_answer = agent.run_turn("Calculate 25 * 4 + 10")

            self.assertEqual(final_answer, "The result of 25 * 4 + 10 is 110.")
            # Verify history contains:
            # 1. user prompt
            # 2. model tool call
            # 3. user function response
            # 4. model final text
            self.assertEqual(len(agent.history), 4)
            self.assertEqual(agent.history[0]["role"], "user")
            self.assertEqual(agent.history[1]["role"], "model")
            self.assertEqual(agent.history[2]["role"], "user")
            self.assertIn("functionResponse", agent.history[2]["parts"][0])
            self.assertEqual(
                agent.history[2]["parts"][0]["functionResponse"]["response"]["result"],
                110
            )
            self.assertEqual(agent.history[3]["role"], "model")


if __name__ == "__main__":
    unittest.main()
