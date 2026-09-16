"""
Parse the knowledge-base markdown files into chunks with preserved metadata.

Each file has YAML front matter (document_id, title, status, policy_authority,
audience, effective_date, supersedes/superseded_by, etc). We split the body on
'##' headings so every chunk keeps a reference to its filename + heading, which
is required for source citations.
"""
from __future__ import annotations

import re
import glob
import os
from dataclasses import dataclass, field
from typing import Any


FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    fm_block, body = m.group(1), m.group(2)
    meta: dict[str, Any] = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    heading: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def source_label(self) -> str:
        return f"{self.filename}#{self.heading}" if self.heading else self.filename


def _split_by_heading(body: str) -> list[tuple[str, str]]:
    """Return list of (heading, section_text). Text before the first '##' is
    kept under the document title ('##' is used for sections; '#' is title)."""
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in lines:
        h2 = re.match(r"^##\s+(.*)", line)
        h1 = re.match(r"^#\s+(.*)", line)
        if h2:
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = h2.group(1).strip()
            current_lines = []
        elif h1:
            # document title line, skip capturing as heading text itself
            continue
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))
    return [(h, "\n".join(t).strip()) for h, t in sections if "\n".join(t).strip()]


def load_knowledge_base(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = _parse_front_matter(raw)
        for i, (heading, text) in enumerate(_split_by_heading(body)):
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}::{i}",
                    filename=filename,
                    heading=heading,
                    text=text,
                    metadata=meta,
                )
            )
    return chunks


if __name__ == "__main__":
    cks = load_knowledge_base(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
    for c in cks[:6]:
        print(c.source_label(), "|", c.metadata.get("status"), "|", c.text[:60].replace("\n", " "))
    print(f"Total chunks: {len(cks)}")
