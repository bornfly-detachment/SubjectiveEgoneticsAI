"""GRPO training launcher for reward-based self-improvement."""
import subprocess, json, logging
from pathlib import Path
from datetime import datetime
from config.settings import settings
from models.training.data_gen import generate_grpo_from_trajectories
from models.training.versioning import register_version, get_active_version

logger = logging.getLogger(__name__)


def run_grpo():
    active = get_active_version()
    base_model = active["checkpoint_path"] if active else settings.model_path
    output_dir = Path(settings.checkpoints_dir) / f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "grpo_data.json"
    samples = generate_grpo_from_trajectories(str(data_path))
    if len(samples) < 10:
        logger.warning("Insufficient GRPO samples, skipping")
        return None

    config = {
        "model_name_or_path": base_model,
        "stage": "grpo",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 32,
        "lora_target": "all",
        "dataset_dir": str(output_dir),
        "dataset": "grpo_data",
        "template": "qwen",
        "output_dir": str(output_dir),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "grpo_beta": 0.04,
        "num_generations": 8,
        "bf16": True,
        "max_steps": 200
    }
    config_path = output_dir / "grpo_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    dataset_info = {"grpo_data": {"file_name": "grpo_data.json", "formatting": "sharegpt"}}
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info))

    reward_avg = sum(s["reward"] for s in samples) / len(samples) if samples else 0
    logger.info(f"Starting GRPO training -> {output_dir}, avg_reward={reward_avg:.3f}")

    result = subprocess.run(
        ["llamafactory-cli", "train", str(config_path)],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        vid = register_version(str(output_dir), "grpo", len(samples),
                               reward_avg=reward_avg, base_version=active["id"] if active else None)
        logger.info(f"GRPO complete. Version: {vid}")
        return vid
    else:
        logger.error(f"GRPO failed: {result.stderr[-2000:]}")
        return None
