"""
POST /llm/chat — 多轮对话接口，支持同步和 SSE 流式输出

请求体:
  messages    : [{"role": "user"|"assistant", "content": "..."}]  必填
  system      : str   可选，系统提示
  model       : str   可选，默认 settings.default_llm_model
  max_tokens  : int   可选，默认 2048
  stream      : bool  可选，默认 false

同步响应: {"text": "...", "usage": {"input_tokens": N, "output_tokens": N}}
流式响应: SSE，data: {"type": "delta"|"done"|"error", "text": "...", "usage": {...}}
"""
import json
import logging
from typing import AsyncIterator

import anthropic
import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    messages: list[dict]
    system: str = ""
    model: str = ""
    max_tokens: int = 2048
    stream: bool = False


def _make_client() -> anthropic.AsyncAnthropic:
    kwargs: dict = {"api_key": settings.anthropic_api_key}
    if settings.anthropic_base_url:
        kwargs["base_url"] = settings.anthropic_base_url
    if settings.llm_proxy:
        kwargs["http_client"] = httpx.AsyncClient(proxy=settings.llm_proxy)
    return anthropic.AsyncAnthropic(**kwargs)


def _build_kwargs(req: ChatRequest) -> dict:
    model = req.model or settings.default_llm_model
    kw: dict = {
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": req.messages,
    }
    if req.system:
        kw["system"] = req.system
    return kw


# ── 同步模式 ──────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(req: ChatRequest):
    if req.stream:
        return StreamingResponse(_stream(req), media_type="text/event-stream")

    try:
        client = _make_client()
        msg = await client.messages.create(**_build_kwargs(req))
        return {
            "text": msg.content[0].text,
            "usage": {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            },
        }
    except Exception as e:
        logger.error(f"[llm/chat] sync error: {e}")
        return {"error": str(e)}


# ── SSE 流式模式 ───────────────────────────────────────────────────────────────

async def _stream(req: ChatRequest) -> AsyncIterator[str]:
    def send(data: dict) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        client = _make_client()
        async with client.messages.stream(**_build_kwargs(req)) as stream:
            async for text in stream.text_stream:
                yield send({"type": "delta", "text": text})
            msg = await stream.get_final_message()
            yield send({
                "type": "done",
                "usage": {
                    "input_tokens": msg.usage.input_tokens,
                    "output_tokens": msg.usage.output_tokens,
                },
            })
    except Exception as e:
        logger.error(f"[llm/chat] stream error: {e}")
        yield send({"type": "error", "error": str(e)})
