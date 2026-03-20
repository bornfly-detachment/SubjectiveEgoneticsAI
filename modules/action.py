"""
Action execution module: LLM API calls + Tool calls.
Called by agent executor for llm_call and tool_call nodes.
Uses anthropic SDK for Anthropic-compatible endpoints (supports custom base_url).
"""
import httpx
from typing import Any
from config.settings import settings


class ActionModule:

    async def llm_call(self, prompt: str, system: str = "", context: dict = None,
                        model: str = None, max_tokens: int = 4000) -> dict:
        """Call external LLM API. Returns {content, input_tokens, output_tokens}."""
        model = model or settings.default_llm_model
        ctx_str = self._format_context(context or {})
        full_prompt = f"{ctx_str}\n\n{prompt}".strip()

        if settings.default_llm_provider == "anthropic":
            return await self._call_anthropic(system, full_prompt, model, max_tokens)
        else:
            return await self._call_openai(system, full_prompt, model, max_tokens)

    async def _call_anthropic(self, system: str, prompt: str, model: str, max_tokens: int) -> dict:
        import anthropic
        import httpx as _httpx

        kwargs = {
            "api_key": settings.anthropic_api_key,
        }
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        # Use explicit proxy for external LLM calls; avoids inheriting HTTP_PROXY for localhost
        if settings.llm_proxy:
            kwargs["http_client"] = _httpx.AsyncClient(proxy=settings.llm_proxy)

        client = anthropic.AsyncAnthropic(**kwargs)

        msg_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            msg_kwargs["system"] = system

        resp = await client.messages.create(**msg_kwargs)
        return {
            "content": resp.content[0].text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    async def _call_openai(self, system: str, prompt: str, model: str, max_tokens: int) -> dict:
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            return {
                "content": choice["message"]["content"],
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"]
            }

    async def tool_call(self, tool_name: str, args: dict, context: dict = None) -> dict:
        """Dispatch to registered tools."""
        tools = {
            "read_file":    self._tool_read_file,
            "write_file":   self._tool_write_file,
            "search_web":   self._tool_search_web,
            "run_python":   self._tool_run_python,
            "claude_code":  self._tool_claude_code,
            "openclaw":     self._tool_openclaw,
        }
        fn = tools.get(tool_name)
        if not fn:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}
        return await fn(args, context or {})

    async def _tool_read_file(self, args: dict, ctx: dict) -> dict:
        path = args.get("path", "")
        try:
            content = open(path).read()
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _tool_write_file(self, args: dict, ctx: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            open(path, "w").write(content)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _tool_search_web(self, args: dict, ctx: dict) -> dict:
        # Placeholder — integrate with actual search API
        return {"ok": False, "error": "search_web not yet configured"}

    async def _tool_run_python(self, args: dict, ctx: dict) -> dict:
        # Sandbox execution placeholder
        return {"ok": False, "error": "run_python not yet enabled (security review needed)"}

    async def _tool_claude_code(self, args: dict, ctx: dict) -> dict:
        from modules.agents import claude_code
        prompt      = args.get("prompt", "")
        working_dir = args.get("working_dir", args.get("cwd", ""))
        timeout     = int(args.get("timeout", 300))
        if not prompt:
            return {"ok": False, "error": "claude_code: prompt is required"}
        return await claude_code.run(prompt=prompt, working_dir=working_dir, timeout=timeout)

    async def _tool_openclaw(self, args: dict, ctx: dict) -> dict:
        from modules.agents import openclaw
        message  = args.get("message", args.get("prompt", ""))
        agent_id = args.get("agent_id")
        timeout  = int(args.get("timeout", 300))
        if not message:
            return {"ok": False, "error": "openclaw: message is required"}
        return await openclaw.run(message=message, agent_id=agent_id, timeout=timeout)

    def _format_context(self, ctx: dict) -> str:
        parts = []
        if ctx.get("variables"):
            parts.append(f"Context variables: {ctx['variables']}")
        if ctx.get("history"):
            recent = ctx["history"][-3:]
            parts.append(f"Recent history: {recent}")
        return "\n".join(parts)
