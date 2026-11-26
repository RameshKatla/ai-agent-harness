import ast
import math
import operator

from pydantic import Field

from app.core.exceptions import ToolInputError
from app.tools.registry import RetryPolicy, Tool, ToolInput

MAX_MAGNITUDE = 10**100
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorInput(ToolInput):
    expression: str = Field(min_length=1, max_length=256)


def _evaluate(node: ast.expr) -> int | float:
    match node:
        case ast.Constant(value=value) if type(value) in (int, float):
            result = value
        case ast.UnaryOp(op=op, operand=operand) if type(op) in UNARY_OPERATORS:
            result = UNARY_OPERATORS[type(op)](_evaluate(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in BINARY_OPERATORS:
            lhs, rhs = _evaluate(left), _evaluate(right)
            if isinstance(op, ast.Pow) and abs(rhs) > 100:
                raise ToolInputError("The exponent cannot exceed 100 in magnitude.")
            result = BINARY_OPERATORS[type(op)](lhs, rhs)
        case _:
            raise ToolInputError("The calculator accepts numeric arithmetic only.")

    if type(result) not in (int, float) or abs(result) > MAX_MAGNITUDE or not math.isfinite(result):
        raise ToolInputError("The calculator result is outside the supported numeric range.")
    return result


async def calculate(arguments: CalculatorInput) -> int | float:
    try:
        return _evaluate(ast.parse(arguments.expression, mode="eval").body)
    except (SyntaxError, ArithmeticError, ValueError):
        raise ToolInputError("Invalid arithmetic expression.") from None


CALCULATOR = Tool(
    name="calculator",
    description="Evaluate numeric arithmetic with +, -, *, /, //, %, ** and parentheses.",
    input_schema=CalculatorInput,
    permission="calculator:use",
    timeout_seconds=1,
    retry_policy=RetryPolicy(),
    implementation=calculate,
)
