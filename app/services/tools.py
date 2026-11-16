from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any

from .knowledge_base import KnowledgeBase, RetrievedChunk


class ToolExecutionError(ValueError):
    pass


@dataclass(slots=True)
class ToolResult:
    content: str
    sources: list[RetrievedChunk]


ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left**right,
}

ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def safe_calculate(expression: str) -> float:
    """Evaluate a small arithmetic expression without using eval()."""
    if not expression or len(expression) > 200:
        raise ToolExecutionError("表达式为空或过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError("表达式格式不正确") from exc
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ToolExecutionError("表达式过于复杂")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ToolExecutionError("指数过大")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ToolExecutionError("不能除以零")
            return ALLOWED_BINARY_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY_OPERATORS:
            return ALLOWED_UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ToolExecutionError("表达式包含不支持的运算")

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ToolExecutionError("计算结果无效")
    return result


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "检索知识库，回答文档相关问题时必须优先调用。返回相关文本片段及引用编号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用于向量检索的完整问题或关键词"},
                    "top_k": {
                        "type": "integer",
                        "description": "返回的文本片段数量，默认 4，最大 8",
                        "minimum": 1,
                        "maximum": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "列出当前知识库中可用的文档。用户询问有哪些资料时使用。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行基础算术运算。仅支持数字和 + - * / // % ** 及括号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "例如 (120+30)*0.8"}
                },
                "required": ["expression"],
            },
        },
    },
]


class DocumentToolbox:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "search_documents":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ToolExecutionError("query 不能为空")
            raw_top_k = arguments.get("top_k", 4)
            try:
                top_k = max(1, min(8, int(raw_top_k)))
            except (TypeError, ValueError):
                top_k = 4
            chunks = await self.knowledge_base.search(query, top_k=top_k + 2)
            chunks = chunks[:top_k]
            if not chunks:
                return ToolResult(content="知识库中没有检索到相关内容。", sources=[])
            payload = [
                {
                    "citation": index,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "content": chunk.text,
                }
                for index, chunk in enumerate(chunks, start=1)
            ]
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False, indent=2),
                sources=chunks,
            )

        if name == "list_documents":
            documents = self.knowledge_base.list_documents()
            payload = [
                {"filename": item.filename, "chunks": item.chunks} for item in documents
            ]
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                sources=[],
            )

        if name == "calculator":
            expression = str(arguments.get("expression", ""))
            result = safe_calculate(expression)
            return ToolResult(
                content=json.dumps({"expression": expression, "result": result}),
                sources=[],
            )

        raise ToolExecutionError(f"未知工具：{name}")
