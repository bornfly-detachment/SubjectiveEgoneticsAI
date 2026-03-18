"""
Local model judgment module: calls inference_service for subjective/value decisions.
"""
import httpx
from config.settings import settings


class JudgeModule:

    async def judge(self, question: str, context: dict = None) -> dict:
        """
        Ask local Qwen2-0.5B for a value judgment.
        Returns {answer, confidence, reasoning}
        """
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"{settings.inference_url}/judge",
                    json={"question": question, "context": context or {}}
                )
                r.raise_for_status()
                return r.json()
        except Exception as e:
            # Fallback: return low confidence to trigger human_gate
            return {
                "answer": "unknown",
                "confidence": 0.0,
                "reasoning": f"Inference service unavailable: {e}"
            }
