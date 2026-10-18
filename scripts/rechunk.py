"""Split long policy chunks so each fits comfortably inside MiniLM's 256-token window.

Usage: python scripts/rechunk.py [source.jsonl] [out.jsonl]
Each output record keeps the parent's metadata, gets id '<parent>#<n>', and an
'embed_text' field ('<section>: <text>') that is what gets embedded.
"""
import json
import re
import sys
from pathlib import Path

MAX_WORDS = 80  # course codes tokenize into several pieces, so stay well under 256 tokens

PROTECT = ["Rs. ", "e.g. ", "i.e. ", "No. ", "etc. "]


def split_sentences(text: str) -> list[str]:
    for p in PROTECT:
        text = text.replace(p, p.replace(" ", " "))
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(\[0-9'\"])", text)
    return [p.replace(" ", " ").strip() for p in parts if p.strip()]


def pack(sentences: list[str], max_words: int) -> list[str]:
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        # a single huge sentence: break it on commas/semicolons/colons
        if w > max_words:
            if cur:
                chunks.append(" ".join(cur)); cur, n = [], 0
            pieces = re.split(r"(?<=[,;:])\s+", s)
            sub, sn = [], 0
            for p in pieces:
                pw = len(p.split())
                if sn + pw > max_words and sub:
                    chunks.append(" ".join(sub)); sub, sn = [], 0
                sub.append(p); sn += pw
            if sub:
                chunks.append(" ".join(sub))
            continue
        if n + w > max_words and cur:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def main(src: Path, out: Path) -> None:
    n_in = n_out = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_in += 1
            pieces = pack(split_sentences(rec["text"]), MAX_WORDS)
            for i, piece in enumerate(pieces, 1):
                new = {k: v for k, v in rec.items() if k != "text"}
                new["parent_id"] = rec["id"]
                new["id"] = f"{rec['id']}#{i}"
                new["part"] = f"{i}/{len(pieces)}"
                new["text"] = piece
                new["embed_text"] = f"{rec['section']}: {piece}"
                g.write(json.dumps(new, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"{n_in} source chunks -> {n_out} retrieval chunks ({out})")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data"
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "source_corpus.jsonl"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else base / "corpus.jsonl"
    main(src, out)
