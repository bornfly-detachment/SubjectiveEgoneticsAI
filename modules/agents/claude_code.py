"""
Claude Code agent wrapper.
Calls `claude` CLI in non-interactive mode via subprocess.
"""
import asyncio
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_BIN = shutil.which("claude") or "claude"


async def run(prompt: str, working_dir: str = None, timeout: int = 300) -> dict:
    """
    Call claude -p <prompt> and return {ok, output, error}.
    working_dir defaults to current directory.
    """
    cwd = Path(working_dir).expanduser() if working_dir else Path.cwd()

    cmd = [CLAUDE_BIN, "-p", prompt, "--output-format", "json"]
    logger.info(f"claude_code: cwd={cwd} prompt={prompt[:80]}...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"Claude Code timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "error": "claude CLI not found. Is Claude Code installed?"}

    out_text = stdout.decode("utf-8", errors="replace").strip()
    err_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        return {"ok": False, "error": err_text or out_text or f"exit code {proc.returncode}"}

    # Try to parse JSON output; fall back to raw text
    try:
        data = json.loads(out_text)
        result_text = data.get("result") or data.get("content") or out_text
    except Exception:
        result_text = out_text

    return {"ok": True, "output": result_text, "raw": out_text}
