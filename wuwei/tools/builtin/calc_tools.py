from __future__ import annotations

import ast
import math
import operator
from typing import Any

from wuwei.tools.registry import ToolRegistry

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MATH_NAMES = {
    name: value
    for name, value in vars(math).items()
    if not name.startswith("_") and (callable(value) or isinstance(value, int | float))
}
_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


def _eval_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp):
        operator_func = _BINARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError("不支持的计算操作")
        return operator_func(_eval_expr(node.left), _eval_expr(node.right))

    if isinstance(node, ast.UnaryOp):
        operator_func = _UNARY_OPERATORS.get(type(node.op))
        if operator_func is None:
            raise ValueError("不支持的一元操作")
        return operator_func(_eval_expr(node.operand))

    if isinstance(node, ast.Name):
        if node.id not in _MATH_NAMES:
            raise ValueError(f"不允许的名称: {node.id}")
        return _MATH_NAMES[node.id]

    if isinstance(node, ast.Call):
        func = _eval_expr(node.func)
        if func not in _MATH_NAMES.values() or not callable(func):
            raise ValueError("只允许调用内置数学函数")
        args = [_eval_expr(arg) for arg in node.args]
        if node.keywords:
            raise ValueError("不支持关键字参数")
        return func(*args)

    raise ValueError("表达式只能包含数字、数学运算和数学函数")


def register_calc_tools(registry: ToolRegistry) -> None:
    @registry.tool(
        name="calculate",
        description=(
            "安全计算数学表达式。支持 + - * / // % **、括号、常用 math 函数和常量，"
            "例如 sqrt(16)、sin(pi / 2)、round(10 / 3, 2)。"
        ),
    )
    def calculate(expression: str) -> dict:
        """计算数学表达式。

        :param expression: 要计算的数学表达式
        """
        parsed = ast.parse(expression, mode="eval")
        result = _eval_expr(parsed)
        return {"ok": True, "expression": expression, "result": result}
