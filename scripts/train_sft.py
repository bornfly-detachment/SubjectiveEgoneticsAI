#!/usr/bin/env python3
"""
SFT 训练启动脚本
用法: ~/llama-factory/venv/bin/python scripts/train_sft.py
"""
import subprocess, sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEAI_DIR = Path(__file__).parent.parent
LLAMA_FACTORY_CLI = Path.home() / "llama-factory/venv/bin/llamafactory-cli"
CONFIG = SEAI_DIR / "configs/sft_qwen3.5.yaml"


def main():
    if not CONFIG.exists():
        logger.error(f"Config not found: {CONFIG}")
        sys.exit(1)

    logger.info(f"Starting SFT training with config: {CONFIG}")
    result = subprocess.run(
        [str(LLAMA_FACTORY_CLI), "train", "--config", str(CONFIG)],
        cwd=str(SEAI_DIR),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
