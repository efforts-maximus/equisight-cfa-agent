"""Permanent Pytest Zero-Trust Anti-Cheat Gate."""

import pathlib
import subprocess
import sys


def test_workstation_zero_trust_honesty_referee():
    """Fails pytest automatically if synthetic calls enter live paths or silent except:pass blocks exist."""
    referee = pathlib.Path.home() / ".gemini" / "audit_honesty.py"
    assert referee.exists(), f"Global referee missing at {referee}"
    res = subprocess.run([sys.executable, str(referee)], capture_output=True, text=True)
    assert res.returncode == 0, f"Zero-Trust Referee Failed:\n{res.stdout}\n{res.stderr}"
