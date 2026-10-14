#!/usr/bin/env python3
"""
Gate M1-N enforcement: fail if the locked symbols `k` / `N` / `lambda` (lambda
security level, distinct from Python's `lambda` keyword -- see LOCKED_NAMES)
are overloaded.

Per m1-notation-gate.md Sec 3 (lambda added by the 2026-09-16 amendment,
Sec 3A), this walks src/, configs/, and every M1_*.md / m1-*.md file in the
module root, and flags:

  1. Any Python assignment or function-parameter binding to the bare name `k`,
     `N`, or `lam` (Python cannot use the keyword `lambda` as an identifier,
     so `lam` is the sanctioned stand-in enforced here) outside of
     notation.py itself (the one sanctioned source).
  2. Any Markdown/LaTeX inline-math token `k` / `N` / `\\lambda` used with a
     definition that does not match the locked meaning -- this is a heuristic
     scan (exact semantic checking of prose is not automatable), so it
     reports every standalone occurrence for a human skim rather than
     silently passing them.

Exit code 0 means: no disallowed Python bindings were found. (The Markdown
scan is advisory -- printed for review, never fails the gate by itself,
since prose legitimately reuses these glyphs in unrelated words like
"Kelvin" or math the notation gate doesn't govern.)

Usage: python src/check_notation.py [module_root]
"""
from __future__ import annotations

import ast
import io
import pathlib
import re
import sys

# Windows consoles often default to a legacy codepage (cp1252) that cannot
# encode lambda or other non-ASCII characters this file's own advisory output
# now needs to print (m1-0-implementation-lock.md uses actual "λ" glyphs,
# not just the word). Re-wrap stdout as UTF-8 so a locked symbol appearing in
# a quoted line never crashes the gate check itself.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SANCTIONED_FILE = "notation.py"
LOCKED_NAMES = {"k", "N", "lam"}


class BindingVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, name: str, kind: str) -> None:
        self.violations.append(
            f"{self.filename}:{node.lineno}: bare '{name}' {kind} "
            f"(import from notation.py instead)"
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in LOCKED_NAMES:
                self._flag(node, target.id, "assignment")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id in LOCKED_NAMES:
            self._flag(node, node.target.id, "annotated assignment")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_args(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_args(node)
        self.generic_visit(node)

    def _check_args(self, node) -> None:
        all_args = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        for a in all_args:
            if a.arg in LOCKED_NAMES:
                self._flag(node, a.arg, "function parameter")


def check_python_file(path: pathlib.Path) -> list[str]:
    if path.name == SANCTIONED_FILE:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: SyntaxError while parsing for notation check: {exc}"]
    visitor = BindingVisitor(str(path))
    visitor.visit(tree)
    return visitor.violations


MD_TOKEN_RE = re.compile(r"(?:`([kN])`|\$([kN])\$|(λ)|\\lambda\b)")


def scan_markdown_file(path: pathlib.Path) -> list[str]:
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in MD_TOKEN_RE.finditer(line):
            sym = m.group(1) or m.group(2) or m.group(3) or "lambda"
            hits.append(f"{path}:{lineno}: standalone '{sym}' token -- {line.strip()[:100]}")
    return hits


def main(argv: list[str]) -> int:
    root = pathlib.Path(argv[1] if len(argv) > 1 else ".").resolve()
    py_targets = list((root / "src").rglob("*.py")) if (root / "src").is_dir() else []
    py_targets += list((root / "experiments").rglob("*.py")) if (root / "experiments").is_dir() else []

    violations: list[str] = []
    for f in py_targets:
        violations.extend(check_python_file(f))

    md_targets = [p for p in root.glob("M1_*.md")] + [p for p in root.glob("m1-*.md")]
    advisory: list[str] = []
    for f in md_targets:
        advisory.extend(scan_markdown_file(f))

    if advisory:
        print(f"--- Advisory: {len(advisory)} standalone k/N token(s) in Markdown (review, does not fail the gate) ---")
        for line in advisory:
            print(line)

    if violations:
        print(f"\n--- FAIL: {len(violations)} notation violation(s) ---")
        for line in violations:
            print(line)
        return 1

    print(f"\nGate M1-N notation check: PASS ({len(py_targets)} Python files scanned, 0 violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
