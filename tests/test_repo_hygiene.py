"""Repository hygiene.

This project's original sin was four live API keys committed to a public repo.
A `.gitignore` was added afterwards — but gitignore does **not** apply to files
git is already tracking, and `.env` stayed tracked. Any `git add -A` would have
re-committed it, this time with real credentials in it.

These tests fail loudly if that regresses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SECRET_FILE_PATTERNS = (".env", ".env.local", "secrets.json", "credentials.json")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout


def _in_git_repo() -> bool:
    return (ROOT / ".git").exists()


pytestmark = pytest.mark.skipif(not _in_git_repo(), reason="not a git repository")


def test_env_file_is_not_tracked():
    """`.env` must never be tracked, regardless of what .gitignore says."""
    tracked = set(_git("ls-files").splitlines())
    offenders = [f for f in tracked if Path(f).name in SECRET_FILE_PATTERNS]
    assert not offenders, (
        f"secret file(s) tracked by git: {offenders}. "
        "Run `git rm --cached <file>` — .gitignore does not untrack existing files."
    )


def test_env_example_contains_no_real_values():
    """The committed template must ship placeholders, not credentials."""
    example = ROOT / ".env.example"
    if not example.exists():
        pytest.skip(".env.example missing")

    suspicious = []
    for line in example.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if not value:
            continue
        # Real credentials from the providers this project uses.
        if value.startswith(("gsk_", "sb_secret_", "sb_publishable_", "sk-", "eyJ")):
            suspicious.append(key)
        if key.endswith(("_KEY", "_SECRET", "_TOKEN", "_PASSWORD")) and len(value) > 24:
            suspicious.append(key)

    assert not suspicious, f".env.example appears to contain real secrets: {suspicious}"
