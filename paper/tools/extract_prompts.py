"""Draft of the prompt log that the AAMAS 2027 AI policy asks for, from Claude Code session transcripts.

Usage: extract_prompts.py TRANSCRIPT.jsonl [...] [--redact WORD ...] [--out paper/ai_prompts_draft.md]
Keeps what the user typed (not tool results, command caveats, system reminders or pasted-content wrappers), in time
order, with the session id and timestamp. The output is a draft for the authors to review, redact and trim before it
goes into the supplementary archive; it is git-ignored.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SKIP = ("<local-command-caveat>", "<command-name>", "<local-command-stdout>", "<system-reminder>",
        "[Request interrupted", "<task-notification>", "Another Claude session sent a message", "[Cross-session",
        "This session is being continued from a previous conversation", "[Image:")


def prompts(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("isSidechain"):
            continue
        att = d.get("attachment") or {}
        if d.get("type") == "attachment" and att.get("type") == "queued_command" and att.get("humanTurn"):
            out.append((att.get("timestamp", d.get("timestamp", "")), str(att.get("prompt", "")).strip()))
            continue  # a message the user typed while the assistant was working
        if d.get("type") != "user":
            continue
        content = d.get("message", {}).get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        if not isinstance(content, str) or not content.strip() or content.lstrip().startswith(SKIP):
            continue
        out.append((d.get("timestamp", ""), content.strip()))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts", type=Path, nargs="+")
    ap.add_argument("--redact", nargs="*", default=[], help="words replaced by [redacted] (host names, user names)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "ai_prompts_draft.md")
    args = ap.parse_args()
    lines = ["# Prompts given to the AI assistant (draft for the authors' review)", "",
             "Tool: Claude Code (Anthropic). One entry per user prompt, in time order.", ""]
    for path in args.transcripts:
        for ts, text in prompts(path):
            for word in args.redact:
                text = re.sub(re.escape(word), "[redacted]", text, flags=re.IGNORECASE)
            lines += [f"## {ts} (session {path.stem[:8]})", "", text, ""]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} ({sum(1 for ln in lines if ln.startswith('## '))} prompts)")


if __name__ == "__main__":
    main()
