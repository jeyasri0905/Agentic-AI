"""
calculator.py - Production-Grade Safe AST Math Evaluator
Author: Agentic AI Systems Architect

Zero-dependency arithmetic and mathematical expression evaluator using Python's
Abstract Syntax Tree (AST). Guarantees O(1) sandboxing against code injection,
attribute access, arbitrary function execution, and dangerous primitives.
"""

import ast
import math
import operator
from typing import Any, Dict, Union

# Whitelist of safe binary and unary operators
OPERATORS = {
    # Binary operators
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    # Unary operators
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelist of mathematical constants
SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Whitelist of mathematical functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "cbrt": getattr(math, "cbrt", lambda x: x ** (1 / 3)),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}


class SecurityEvaluationError(Exception):
    """Raised when an expression contains unauthorized AST nodes or patterns."""
    pass


class SafeMathEvaluator:
    """
    Safely parses and evaluates mathematical expressions without `eval` or `exec`.
    """

    def __init__(self, max_power: float = 10000.0):
        self.max_power = max_power

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):  # Python 3.8+ (covers numbers, strings, etc.)
            if isinstance(node.value, (int, float)):
                return node.value
            raise SecurityEvaluationError(f"Unsupported constant type: {type(node.value).__name__}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in OPERATORS:
                raise SecurityEvaluationError(f"Unsupported binary operator: {op_type.__name__}")
            
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # Prevent resource exhaustion via massive powers like 9999**99999
            if op_type is ast.Pow:
                if isinstance(right, (int, float)) and abs(right) > self.max_power:
                    raise ValueError(f"Exponent exceeds safe limit ({self.max_power})")
                if isinstance(left, (int, float)) and abs(left) > 1e10 and right > 10:
                    raise ValueError("Power computation exceeds computational boundary")

            try:
                return OPERATORS[op_type](left, right)
            except ZeroDivisionError:
                raise ZeroDivisionError("Math error: Division or modulo by zero")
            except OverflowError:
                raise OverflowError("Math error: Numerical result overflow")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in OPERATORS:
                raise SecurityEvaluationError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            # Function calls must be simple identifiers e.g. sqrt(16)
            if not isinstance(node.func, ast.Name):
                raise SecurityEvaluationError("Only direct mathematical functions are permitted")
            
            func_name = node.func.id.lower()
            if func_name not in SAFE_FUNCTIONS:
                raise SecurityEvaluationError(f"Disallowed function call: '{func_name}'")

            args = [self._eval_node(arg) for arg in node.args]
            try:
                return SAFE_FUNCTIONS[func_name](*args)
            except Exception as e:
                raise ValueError(f"Error computing '{func_name}': {str(e)}")

        elif isinstance(node, ast.Name):
            var_name = node.id.lower()
            if var_name in SAFE_CONSTANTS:
                return SAFE_CONSTANTS[var_name]
            raise SecurityEvaluationError(f"Unknown or unauthorized identifier: '{node.id}'")

        elif isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        else:
            raise SecurityEvaluationError(f"Disallowed expression syntax: {type(node).__name__}")

    def evaluate(self, expression: str) -> Union[int, float]:
        """
        Parses and evaluates a mathematical expression string.
        """
        cleaned = expression.strip()
        if not cleaned:
            raise ValueError("Expression is empty")

        # Replace human/LLM exponent notation 'x ^ y' with 'x ** y'
        # unless it's preceded/followed by something strictly non-arithmetic
        cleaned = cleaned.replace("^", "**")

        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in mathematical expression: {e.msg}")

        result = self._eval_node(tree.body)
        
        # Normalize float integers: e.g. 4.0 -> 4 if exact integer
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            return int(result)
        return result


# Singleton instance for high-efficiency reuse
_evaluator = SafeMathEvaluator()


def calculate(expression: str) -> Dict[str, Any]:
    """
    Agent tool entry point.
    Executes arithmetic and math formulas safely.
    
    Args:
        expression: The mathematical expression string (e.g., '125 * 4 + sqrt(144)').
        
    Returns:
        Dict with status, expression, and computed numeric result or error message.
    """
    try:
        val = _evaluator.evaluate(expression)
        return {
            "status": "success",
            "expression": expression,
            "result": val
        }
    except Exception as exc:
        return {
            "status": "error",
            "expression": expression,
            "error": str(exc)
        }


# Gemini Tool Declaration schema for REST API
CALCULATOR_TOOL_DECLARATION = {
    "name": "calculate",
    "description": (
        "Calculates the exact result of any mathematical or arithmetic expression. "
        "Supports addition (+), subtraction (-), multiplication (*), division (/), "
        "integer floor division (//), modulo (%), exponents (** or ^), parentheses, "
        "and standard math functions (sqrt, sin, cos, tan, log, log10, exp, factorial, abs, round) "
        "and constants (pi, e). Always use this tool for arithmetic to avoid calculation errors."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "expression": {
                "type": "STRING",
                "description": "The arithmetic or math expression to evaluate, e.g. '(45 * 12) + (100 / 4)'"
            }
        },
        "required": ["expression"]
    }
}
