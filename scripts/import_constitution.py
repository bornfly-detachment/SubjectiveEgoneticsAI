"""
import_constitution.py
Notion Markdown → SFT training data for local judge model.

Generates two output files:
  data/constitution/sft_alpaca.json    — LlamaFactory alpaca format (instruction tuning)
  data/constitution/sft_judge.json     — Judge-specific Q&A format for /judge endpoint

Usage:
  cd /Users/bornfly/Desktop/SubjectiveEgoneticsAI
  python scripts/import_constitution.py [--notion-dir PATH] [--dry-run]
"""

import re
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

NOTION_DIR = Path("/Users/bornfly/Desktop/bornfly_notion_data")
OUT_DIR    = Path(__file__).parent.parent / "data" / "constitution"

# ── 清理 Notion 特有的噪声 ────────────────────────────────────────────────────

UUID_PATTERN = re.compile(r'\s+[0-9a-f]{32}')      # 文件名末尾 UUID
NOTION_PROP  = re.compile(r'^[^\n]+:\s+\w.*\n', re.MULTILINE)  # Property: value 行

def clean_md(text: str) -> str:
    """Strip Notion metadata noise, keep human-readable content."""
    # Remove blank notion property lines at top
    lines = text.splitlines()
    # Skip leading property-style lines (e.g. "Created: 2024-01-01")
    content_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#') or (line.strip() and not re.match(r'^[\w ]+:\s+\S', line)):
            content_start = i
            break
    text = '\n'.join(lines[content_start:])
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


def extract_title(text: str) -> str:
    m = re.search(r'^#\s+(.+)', text, re.MULTILINE)
    return m.group(1).strip() if m else ''


def is_toc_file(text: str) -> bool:
    """Return True if the file is mostly Notion link index (no real content)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    link_lines = sum(1 for l in lines if l.startswith('[') and '](http' not in l and '%' in l)
    return link_lines / max(len(lines), 1) > 0.4


def strip_notion_links(text: str) -> str:
    """Remove Notion-style URL-encoded markdown links, keep link text."""
    # [label](url-encoded-path) → label
    text = re.sub(r'\[([^\]]+)\]\([^)]*%[^)]*\)', r'\1', text)
    return text


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split by ## headings → list of (heading, body)."""
    parts = re.split(r'\n#{1,3} ', '\n' + text)
    sections = []
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].strip().lstrip('#').strip()
        body = '\n'.join(lines[1:]).strip()
        body = strip_notion_links(body)
        if heading and body and len(body) > 30:
            sections.append((heading, body))
    # Also add full doc as one section if no headings
    if not sections and len(text) > 50:
        title = extract_title(text) or '原则'
        body = re.sub(r'^#[^\n]*\n', '', text).strip()
        body = strip_notion_links(body)
        if body and len(body) > 30:
            sections.append((title, body))
    return sections


# ── SFT 样本生成策略 ──────────────────────────────────────────────────────────

def make_alpaca_samples(title: str, heading: str, body: str) -> list[dict]:
    """Generate multiple alpaca-format samples from one section."""
    samples = []
    body_short = body[:600]  # keep output focused

    # 1. 直接理解：解释原则
    samples.append({
        "instruction": f"请根据生变论解释「{heading}」的含义。",
        "input": "",
        "output": body_short
    })

    # 2. 应用判断：如何看待某事
    samples.append({
        "instruction": f"从生变论的视角，如何理解和践行「{heading}」？",
        "input": f"文档来源：{title}",
        "output": body_short
    })

    # 3. 价值判断：提取原则句子 → 是否认同
    principle_lines = [l.strip() for l in body.splitlines()
                       if l.strip() and len(l.strip()) > 15 and not l.startswith('#')]
    for line in principle_lines[:2]:
        samples.append({
            "instruction": f"你认同以下观点吗？请从生变论出发做出判断并给出理由：\n{line}",
            "input": "",
            "output": f"认同。这符合生变论的核心原则。{body_short[:200]}"
        })

    return samples


def make_judge_samples(title: str, heading: str, body: str) -> list[dict]:
    """Generate /judge endpoint format: question → {answer, confidence, reasoning}."""
    samples = []
    body_short = body[:400]

    questions = [
        f"「{heading}」是否符合生变论的核心价值观？",
        f"「{heading}」这个原则是否应该被遵守？",
    ]

    for q in questions:
        samples.append({
            "instruction": "你是一个自我控制论判断系统。根据宪法原则对问题做出主观判断。输出格式（JSON）：{\"answer\": \"是/否/不确定\", \"confidence\": 0.0-1.0, \"reasoning\": \"判断依据\"}",
            "input": f"问题：{q}\n原则参考：{body_short}",
            "output": json.dumps({
                "answer": "是",
                "confidence": 0.85,
                "reasoning": f"根据生变论，{heading}是核心原则之一。{body_short[:150]}"
            }, ensure_ascii=False)
        })

    return samples


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def collect_md_files(notion_dir: Path, focus_dirs: list[str] = None) -> list[Path]:
    """Collect markdown files, optionally filtering by directory names."""
    all_files = list(notion_dir.rglob("*.md"))
    if not focus_dirs:
        return all_files
    filtered = []
    for f in all_files:
        for d in focus_dirs:
            if d in str(f):
                filtered.append(f)
                break
    return filtered


def process_file(md_path: Path) -> tuple[list[dict], list[dict]]:
    """Process one markdown file → (alpaca_samples, judge_samples)."""
    try:
        raw = md_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        log.warning(f"Cannot read {md_path}: {e}")
        return [], []

    text = clean_md(raw)
    if len(text) < 50:
        return [], []
    if is_toc_file(text):
        return [], []

    title = extract_title(text) or md_path.stem
    # Clean UUID from title
    title = UUID_PATTERN.sub('', title).strip()

    sections = split_sections(text)
    if not sections:
        return [], []

    alpaca, judge = [], []
    for heading, body in sections:
        alpaca.extend(make_alpaca_samples(title, heading, body))
        judge.extend(make_judge_samples(title, heading, body))

    return alpaca, judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notion-dir', default=str(NOTION_DIR))
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    parser.add_argument('--dry-run', action='store_true')
    # Focus on constitution-relevant directories by default
    parser.add_argument('--focus', nargs='*', default=[
        '生变论', '自我控制论', '第一性原理', '生变录', '生命三大定律'
    ])
    args = parser.parse_args()

    notion_dir = Path(args.notion_dir)
    out_dir    = Path(args.out_dir)

    if not notion_dir.exists():
        log.error(f"Notion dir not found: {notion_dir}")
        return

    log.info(f"Scanning: {notion_dir}")
    files = collect_md_files(notion_dir, args.focus if args.focus else None)
    log.info(f"Found {len(files)} markdown files (focus: {args.focus})")

    all_alpaca, all_judge = [], []
    for f in files:
        a, j = process_file(f)
        all_alpaca.extend(a)
        all_judge.extend(j)
        if a:
            log.info(f"  {f.name[:60]} → {len(a)} alpaca, {len(j)} judge samples")

    # Deduplicate by instruction + input combined
    def dedup(samples: list[dict]) -> list[dict]:
        seen, out = set(), []
        for s in samples:
            key = s.get('instruction', '')[:60] + '|' + s.get('input', '')[:60]
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    all_alpaca = dedup(all_alpaca)
    all_judge  = dedup(all_judge)

    log.info(f"\nTotal: {len(all_alpaca)} alpaca samples, {len(all_judge)} judge samples")

    if args.dry_run:
        log.info("Dry run — not writing files")
        # Show first 2 samples
        if all_alpaca:
            print("\n=== Sample alpaca ===")
            print(json.dumps(all_alpaca[0], ensure_ascii=False, indent=2))
        if all_judge:
            print("\n=== Sample judge ===")
            print(json.dumps(all_judge[0], ensure_ascii=False, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_path = out_dir / "sft_alpaca.json"
    judge_path  = out_dir / "sft_judge.json"
    manifest_path = out_dir / "manifest.json"

    alpaca_path.write_text(json.dumps(all_alpaca, ensure_ascii=False, indent=2))
    judge_path.write_text(json.dumps(all_judge, ensure_ascii=False, indent=2))
    manifest_path.write_text(json.dumps({
        "generated_at": str(Path(__file__).stat().st_mtime),
        "notion_dir": str(notion_dir),
        "files_processed": len(files),
        "alpaca_samples": len(all_alpaca),
        "judge_samples": len(all_judge),
        "focus_dirs": args.focus
    }, ensure_ascii=False, indent=2))

    log.info(f"Written: {alpaca_path}")
    log.info(f"Written: {judge_path}")
    log.info(f"Written: {manifest_path}")


if __name__ == '__main__':
    main()
