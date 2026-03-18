"""
Local model inference service. FastAPI on port 8001.
Run: uvicorn models.inference:app --port 8001
"""
import json, logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_model = None
_tokenizer = None

def load_model(path: str = None):
    global _model, _tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from config.settings import settings
    model_path = path or settings.model_path
    logger.info(f"Loading model from {model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto", trust_remote_code=True
    )
    _model.eval()
    logger.info("Model loaded")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="SubjectiveEgoneticsAI Inference", version="0.1.0", lifespan=lifespan)

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

@app.post("/judge", response_model=JudgeResponse)
async def judge(req: JudgeRequest):
    global _model, _tokenizer
    if _model is None:
        load_model()
    import torch
    ctx_str = json.dumps(req.context, ensure_ascii=False) if req.context else ""
    prompt = f"问题：{req.question}\n上下文：{ctx_str}"
    if req.constitution_hint:
        prompt += f"\n原则参考：{req.constitution_hint}"
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=256, temperature=0.1,
                                  do_sample=False, pad_token_id=_tokenizer.eos_token_id)
    raw = _tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        data = json.loads(raw.strip())
        return JudgeResponse(answer=data.get("answer","不确定"),
                             confidence=float(data.get("confidence",0.3)),
                             reasoning=data.get("reasoning",""), raw_output=raw)
    except Exception:
        return JudgeResponse(answer="不确定", confidence=0.1, reasoning="解析失败", raw_output=raw)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/reload-model")
async def reload_model(checkpoint_path: str):
    global _model, _tokenizer
    _model = _tokenizer = None
    load_model(checkpoint_path)
    return {"status": "reloaded", "checkpoint": checkpoint_path}
