#!/usr/bin/env python
"""Release hygiene checks for the public Qwen3-8B quantization repository."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
ALLOW_LARGE = {
    Path("assets/figures/mlp_residual_mechanism.png"),
}

FORBIDDEN_NAMES = {".DS_Store"}
FORBIDDEN_PATTERNS = [
    re.compile("/" + "root"),
    re.compile("/" + "root" + "/" + "autodl-tmp"),
    re.compile("root" + "@"),
    re.compile("connect" + r"\.[A-Za-z0-9.-]*" + "seeta" + "cloud" + r"\.com"),
    re.compile("127" + r"\.0\.0\.1" + ":7897"),
    re.compile("github" + r"_pat_[A-Za-z0-9_]+"),
    re.compile("ghp" + r"_[A-Za-z0-9_]+"),
    re.compile("hf" + r"_[A-Za-z0-9]+"),
    re.compile(r"BEGIN (RSA|OPENSSH|PRIVATE) KEY"),
]

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def iter_files() -> list[Path]:
    ignored_dirs = {".git", ".pytest_cache", "__pycache__"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def check_forbidden_files(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden file: {rel}")
        if path.name.startswith("samples_") and path.suffix == ".jsonl":
            errors.append(f"raw sample file should not be committed: {rel}")
        if path.suffix == ".tgz":
            errors.append(f"archive file should not be committed: {rel}")
        if path.stat().st_size > MAX_BYTES and rel not in ALLOW_LARGE:
            errors.append(f"large file exceeds 5 MiB: {rel}")
    return errors


def check_sensitive_text(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"sensitive pattern {pattern.pattern!r} in {rel}")
    return errors


def markdown_links(markdown: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", markdown)


def check_markdown_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix != ".md":
            continue
        base = path.parent
        for link in markdown_links(path.read_text(encoding="utf-8", errors="ignore")):
            if "://" in link or link.startswith("#") or link.startswith("mailto:"):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            if not (base / target).exists():
                errors.append(f"broken markdown link in {path.relative_to(ROOT)}: {link}")
    return errors


def main() -> int:
    os.chdir(ROOT)
    files = iter_files()
    errors = []
    errors.extend(check_forbidden_files(files))
    errors.extend(check_sensitive_text(files))
    errors.extend(check_markdown_links(files))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"release hygiene checks passed for {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
