"""
OpenClaw agent wrapper.
Calls `openclaw agent --message <msg> --json` via subprocess.
"""
import asyncio
import json
import logging
import shutil

logger = logging.getLogger(__name__)

OPENCLAW_BIN = shutil.which("openclaw") or "openclaw"


async def run(message: str, agent_id: str = None, timeout: int = 300) -> dict:
    """
    Call openclaw agent and return {ok, output, error}.
    agent_id: optional specific agent to target (openclaw --agent <id>)
    """
    cmd = [OPENCLAW_BIN, "agent", "--message", message, "--json"]
    if agent_id:
        cmd += ["--agent", agent_id]

    logger.info(f"openclaw: agent={agent_id or 'default'} message={message[:80]}...")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"OpenClaw timed out after {timeout}s"}
    except FileNotFoundError:
        return {"ok": False, "error": "openclaw CLI not found"}

    out_text = stdout.decode("utf-8", errors="replace").strip()
    err_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        return {"ok": False, "error": err_text or out_text or f"exit code {proc.returncode}"}

    try:
        data = json.loads(out_text)
        # openclaw --json wraps response in {reply: ..., session_id: ...}
        result_text = data.get("reply") or data.get("content") or data.get("result") or out_text
        return {"ok": True, "output": result_text, "raw": out_text, "session_id": data.get("session_id")}
    except Exception:
        return {"ok": True, "output": out_text}
