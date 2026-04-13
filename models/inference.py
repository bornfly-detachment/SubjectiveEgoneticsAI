"""
Local model inference service — MLX backend (Apple Silicon optimized).
Run: uvicorn models.inference:app --port 8001
"""
import json, logging, time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_model = None
_tokenizer = None

def load_model(path: str = None):
    global _model, _tokenizer
    import mlx_lm
    from config.settings import settings
    model_path = path or settings.model_path
    logger.info(f"Loading model (MLX) from {model_path}")
    _model, _tokenizer = mlx_lm.load(model_path)
    # warmup: eliminate JIT compilation latency on first real request
    _warmup()
    logger.info("Model loaded and warmed up")

def _warmup():
    import mlx_lm
    prompt = _tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False, add_generation_prompt=True
    )
    mlx_lm.generate(_model, _tokenizer, prompt=prompt, max_tokens=8, verbose=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="SubjectiveEgoneticsAI Inference", version="0.2.0", lifespan=lifespan)

JUDGE_SYSTEM = """你是一个自我控制论判断系统。根据用户的宪法原则和价值观，对问题做出主观判断。
输出格式（JSON）：{"answer": "是/否/不确定", "confidence": 0.0-1.0, "reasoning": "判断依据"}
只输出JSON，不要其他内容。"""

class JudgeRequest(BaseModel):
    question: str
    context: dict = {}
    constitution_hint: Optional[str] = None

class JudgeResponse(BaseModel):
    answer: str
    confidence: float
    reasoning: str
    raw_output: str
    tokens_per_second: float = 0.0

class GenerateRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    tokens_per_second: float

@app.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest):
    import mlx_lm
    ctx_str = json.dumps(req.context, ensure_ascii=False) if req.context else ""
    user_content = f"问题：{req.question}\n上下文：{ctx_str}"
    if req.constitution_hint:
        user_content += f"\n原则参考：{req.constitution_hint}"
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user_content}]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from mlx_lm.sample_utils import make_sampler
    t0 = time.time()
    raw = mlx_lm.generate(_model, _tokenizer, prompt=prompt, max_tokens=256,
                          sampler=make_sampler(temp=0.1), verbose=False)
    elapsed = time.time() - t0
    n_tokens = len(_tokenizer.encode(raw))
    tps = round(n_tokens / elapsed, 1) if elapsed > 0 else 0.0

    try:
        data = json.loads(raw.strip())
        return JudgeResponse(
            answer=data.get("answer", "不确定"),
            confidence=float(data.get("confidence", 0.3)),
            reasoning=data.get("reasoning", ""),
            raw_output=raw,
            tokens_per_second=tps,
        )
    except Exception:
        return JudgeResponse(answer="不确定", confidence=0.1, reasoning="解析失败",
                             raw_output=raw, tokens_per_second=tps)

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    import mlx_lm
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    from mlx_lm.sample_utils import make_sampler
    t0 = time.time()
    text = mlx_lm.generate(_model, _tokenizer, prompt=prompt,
                            max_tokens=req.max_tokens,
                            sampler=make_sampler(temp=req.temperature), verbose=False)
    elapsed = time.time() - t0
    tps = round(len(_tokenizer.encode(text)) / elapsed, 1) if elapsed > 0 else 0.0
    return GenerateResponse(text=text, tokens_per_second=tps)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "backend": "mlx"}

@app.post("/reload-model")
async def reload_model(checkpoint_path: str):
    global _model, _tokenizer
    _model = _tokenizer = None
    load_model(checkpoint_path)
    return {"status": "reloaded", "checkpoint": checkpoint_path}
