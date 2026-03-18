"""
Environment feedback module: evaluates execution results, determines completion.
"""
from typing import Any
from modules.action import ActionModule


class EnvFeedbackModule:

    def __init__(self, action: ActionModule):
        self.action = action

    async def evaluate(self, goal: str, result: Any, context: dict = None) -> dict:
        """Evaluate if the result meets the goal. Returns completion assessment."""
        prompt = f"""Evaluate if this result meets the goal.

Goal: {goal}
Result: {result}

Respond with JSON:
{{
  "completion_rate": 0.0-1.0,
  "quality_score": 0.0-1.0,
  "issues": ["..."],
  "next_hint": "suggestion for next step if incomplete"
}}"""
        response = await self.action.llm_call(prompt=prompt, context=context)
        import json
        try:
            return json.loads(response["content"])
        except Exception:
            return {"completion_rate": 0.5, "quality_score": 0.5, "issues": [], "next_hint": ""}
