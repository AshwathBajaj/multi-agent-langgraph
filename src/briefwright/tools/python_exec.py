"""Restricted Python tool. A demo boundary, not a multi-tenant sandbox."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass

BLOCKED_CALLS = frozenset(
    {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "help",
        "dir",
        "memoryview",
        "classmethod",
        "staticmethod",
        "super",
        "type",
        "object",
        "property",
    }
)

_RUNNER = r"""
import collections
import json
import math
import statistics
import sys

payload = json.loads(sys.stdin.read())
safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "map": map,
    "filter": filter,
    "isinstance": isinstance,
}
globals_dict = {
    "__builtins__": safe_builtins,
    "math": math,
    "statistics": statistics,
    "json": json,
    "collections": collections,
    "DATA": payload["data"],
}
exec(payload["code"], globals_dict, None)
"""


class UnsafeCode(ValueError):
    """The analyst code failed the static check."""


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    error: str = ""


def validate_code(code: str) -> None:
    if not code.strip():
        raise UnsafeCode("Code is empty.")
    if len(code) > 4000:
        raise UnsafeCode("Code is longer than 4000 characters.")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCode(f"Syntax error: {exc.msg}") from exc
    _SafetyVisitor().visit(tree)


def run_python(code: str, data: dict | None = None, timeout: float = 2.0) -> ExecResult:
    try:
        validate_code(code)
    except UnsafeCode as exc:
        return ExecResult(ok=False, stdout="", stderr="", error=str(exc))
    payload = json.dumps({"code": code, "data": data or {}})
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, stdout="", stderr="", error=f"Timed out after {timeout:.1f}s.")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Python tool failed.").strip()
        return ExecResult(ok=False, stdout=completed.stdout, stderr=completed.stderr, error=detail[-800:])
    return ExecResult(ok=True, stdout=completed.stdout, stderr=completed.stderr, error="")


class _SafetyVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import) -> None:
        raise UnsafeCode("Imports are disabled. math, statistics, json, and collections are already loaded.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise UnsafeCode("Imports are disabled. math, statistics, json, and collections are already loaded.")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            raise UnsafeCode(f"Call to {node.func.id} is blocked.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise UnsafeCode("Dunder attribute access is blocked.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise UnsafeCode("Dunder names are blocked.")
        self.generic_visit(node)
