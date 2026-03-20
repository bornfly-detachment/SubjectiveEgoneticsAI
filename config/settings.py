from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    egonetics_url: str = "http://localhost:3002"
    egonetics_token: str = ""

    model_path: str = "/Users/bornfly/Desktop/qwen-edge-llm/model_weights/Qwen/Qwen2-0.5B-Instruct"
    inference_port: int = 8001
    api_port: int = 8000
    inference_url: str = "http://localhost:8001"

    anthropic_api_key: str = ""
    anthropic_base_url: str = ""   # override; leave empty for official API
    openai_api_key: str = ""
    default_llm_provider: str = "anthropic"  # or "openai"
    default_llm_model: str = "ark-code-latest"
    llm_proxy: str = ""            # explicit HTTP proxy for LLM API calls (e.g. http://127.0.0.1:15369)

    db_path: str = str(BASE_DIR / "store" / "exec.db")
    checkpoints_dir: str = str(BASE_DIR / "data" / "checkpoints")
    constitution_dir: str = str(BASE_DIR / "data" / "constitution")
    trajectories_dir: str = str(BASE_DIR / "data" / "trajectories")

    # Agent loop
    judge_confidence_threshold: float = 0.6  # below this → ask user
    max_backtrack_depth: int = 5
    node_timeout_seconds: int = 300
    max_loop_detection_count: int = 3

    # Training triggers
    grpo_trigger_count: int = 50   # new trajectories before GRPO run
    sft_trigger_count: int = 20    # new feedback samples before SFT run

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
