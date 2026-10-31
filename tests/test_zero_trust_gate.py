"""Permanent Pytest Zero-Trust Anti-Cheat Gate (Cross-Platform Portable)."""

import ast
import pathlib
import subprocess
import sys


def _builtin_ast_honesty_check(repo_root: pathlib.Path) -> list[str]:
    """Self-contained AST check for external machines without ~/.gemini/audit_honesty.py."""
    violations: list[str] = []
    scan_files = list((repo_root / "src").rglob("*.py"))
    for entry in ["app.py", "main.py"]:
        p = repo_root / entry
        if p.exists():
            scan_files.append(p)

    for py_file in sorted(scan_files):
        rel = py_file.relative_to(repo_root)
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"), filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if isinstance(child, ast.ExceptHandler):
                        if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                            violations.append(f"{rel}:{child.lineno} in '{node.name}()' -> Silent 'except: pass'")
    return violations


def test_workstation_zero_trust_honesty_referee():
    """Fails pytest automatically if synthetic calls enter live paths or silent except:pass blocks exist."""
    referee = pathlib.Path.home() / ".gemini" / "audit_honesty.py"
    if referee.exists():
        res = subprocess.run([sys.executable, str(referee)], capture_output=True, text=True)
        assert res.returncode == 0, f"Zero-Trust Referee Failed:\n{res.stdout}\n{res.stderr}"
    else:
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        issues = _builtin_ast_honesty_check(repo_root)
        assert not issues, "Zero-Trust AST violations detected:\n" + "\n".join(issues)
