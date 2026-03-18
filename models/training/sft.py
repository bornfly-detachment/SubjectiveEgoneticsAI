"""SFT training launcher using LlamaFactory CLI."""
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime
from config.settings import settings
from models.training.data_gen import generate_sft_from_feedback
from models.training.versioning import register_version

logger = logging.getLogger(__name__)


def run_sft():
    output_dir = Path(settings.checkpoints_dir) / f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate latest training data
    data_path = output_dir / "sft_data.json"
    samples = generate_sft_from_feedback(str(data_path))
    if not samples:
        logger.warning("No SFT samples available, skipping training")
        return None

    config = {
        "model_name_or_path": settings.model_path,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 32,
        "lora_target": "all",
        "dataset_dir": str(output_dir),
        "dataset": "sft_data",
        "template": "qwen",
        "output_dir": str(output_dir),
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "save_steps": 50,
        "logging_steps": 10
    }

    config_path = output_dir / "sft_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    # Register dataset
    dataset_info = {"sft_data": {"file_name": "sft_data.json", "formatting": "alpaca"}}
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info))

    logger.info(f"Starting SFT training -> {output_dir}")
    result = subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        vid = register_version(str(output_dir), "sft", len(samples))
        logger.info(f"SFT complete. Version: {vid}")
        return vid
    else:
        logger.error(f"SFT failed: {result.stderr[-2000:]}")
        return None
