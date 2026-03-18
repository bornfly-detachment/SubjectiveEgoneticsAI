"""Model version management endpoints."""
from fastapi import APIRouter, BackgroundTasks
from models.training.versioning import list_versions, activate_version, get_active_version
from models.training.data_gen import should_trigger_sft, should_trigger_grpo

router = APIRouter()


@router.get("/versions")
async def versions():
    return list_versions()


@router.get("/active")
async def active():
    return get_active_version()


@router.post("/activate/{version_id}")
async def activate(version_id: str):
    ver = activate_version(version_id)
    # Hot-swap in inference service
    import httpx
    from config.settings import settings
    try:
        async with httpx.AsyncClient() as c:
            await c.post(f"{settings.inference_url}/reload-model",
                         params={"checkpoint_path": ver["checkpoint_path"]})
    except Exception:
        pass
    return ver


@router.get("/training-status")
async def training_status():
    return {
        "sft_ready": should_trigger_sft(),
        "grpo_ready": should_trigger_grpo()
    }


@router.post("/train/sft")
async def trigger_sft(bg: BackgroundTasks):
    from models.training.sft import run_sft
    bg.add_task(run_sft)
    return {"status": "sft training started"}


@router.post("/train/grpo")
async def trigger_grpo(bg: BackgroundTasks):
    from models.training.grpo import run_grpo
    bg.add_task(run_grpo)
    return {"status": "grpo training started"}
