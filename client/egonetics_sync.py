"""
egonetics_sync.py — SEAI → Egonetics 实时推送客户端

职责：
  SEAI 的任何状态写入，不再直接调 REST，而是通过 WebSocket propose 到
  Egonetics 消息队列，由用户在前端裁决后持久化。

用法：
  from client.egonetics_sync import sync

  # 提议更新某实体的三问分类（可能有冲突，由用户裁决）
  await sync.propose_classification(
      entity_id="task-123",
      entity_type="task_component",
      layer="P",
      from_tags=["tag-001"],
      what_tags=["tag-002"],
      where_tags=[],
      description="感知层输入信号处理",
      message="根据生命周期状态自动打标",
  )

  # 提议更新任务状态（无冲突时自动应用）
  await sync.propose_task_update(
      task_id="t-xxx",
      patch={"column_id": "done", "task_summary": "已完成"},
      message="任务执行完毕",
  )
"""

import asyncio
import json
import logging
import random
import string
import time
from typing import Any

import websockets
from config.settings import settings

logger = logging.getLogger(__name__)


def _gen_id(prefix: str = "seai") -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}-{int(time.time() * 1000)}-{suffix}"


class EgoneticsSync:
    """
    WebSocket 长连接客户端。
    自动重连，发送 proposal，接收回执。
    """

    def __init__(self):
        self._ws_url = f"ws://{settings.egonetics_host}/ws/seai"
        self._ws = None
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._recv_task = None

    # ── 连接管理 ────────────────────────────────────────────

    async def connect(self):
        """建立 WebSocket 连接（支持断线重连）。"""
        if self._ws and not self._ws.closed:
            return
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                ping_interval=20,
                ping_timeout=10,
            )
            self._running = True
            self._recv_task = asyncio.create_task(self._recv_loop())
            logger.info(f"[sync] connected to {self._ws_url}")
        except Exception as e:
            logger.warning(f"[sync] connect failed: {e}")
            self._ws = None

    async def disconnect(self):
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()

    async def _recv_loop(self):
        """接收服务端回执，解析并 resolve 对应 Future。"""
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    prop_id = data.get("id")
                    if prop_id and prop_id in self._pending:
                        fut = self._pending.pop(prop_id)
                        if not fut.done():
                            fut.set_result(data)
                except Exception:
                    pass
        except Exception as e:
            if self._running:
                logger.warning(f"[sync] recv loop error: {e}, reconnecting…")
                await asyncio.sleep(2)
                await self.connect()

    # ── 核心发送 ─────────────────────────────────────────────

    async def _send(self, msg: dict, timeout: float = 10.0) -> dict:
        """
        发送一条 proposal，等待服务端回执。
        连接断开时自动重连一次。
        """
        async with self._lock:
            if not self._ws or self._ws.closed:
                await self.connect()

        if not self._ws:
            logger.error("[sync] no connection, proposal dropped")
            return {"ok": False, "error": "no connection"}

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        prop_id = msg.get("id") or _gen_id()
        msg["id"] = prop_id
        self._pending[prop_id] = fut

        try:
            await self._ws.send(json.dumps(msg))
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(prop_id, None)
            logger.warning(f"[sync] proposal {prop_id} timed out")
            return {"ok": False, "id": prop_id, "error": "timeout"}
        except Exception as e:
            self._pending.pop(prop_id, None)
            logger.error(f"[sync] send error: {e}")
            return {"ok": False, "error": str(e)}

    # ── 高层语义接口 ──────────────────────────────────────────

    async def propose_classification(
        self,
        entity_id: str,
        entity_type: str,
        layer: str = "",
        from_tags: list[str] = None,
        what_tags: list[str] = None,
        where_tags: list[str] = None,
        description: str = "",
        message: str = "",
    ) -> dict:
        """提议更新某实体的三问分类标签。"""
        return await self._send({
            "type": "classification",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "payload": {
                "layer": layer,
                "from_tags":  from_tags  or [],
                "what_tags":  what_tags  or [],
                "where_tags": where_tags or [],
                "description": description,
            },
            "message": message,
        })

    async def propose_task_update(
        self,
        task_id: str,
        patch: dict[str, Any],
        message: str = "",
    ) -> dict:
        """提议更新任务字段（column_id, task_summary 等）。"""
        return await self._send({
            "type": "task",
            "entity_id": task_id,
            "entity_type": "task",
            "payload": patch,
            "message": message,
        })

    async def propose_tag_tree_change(
        self,
        node_id: str,
        patch: dict[str, Any],
        message: str = "",
    ) -> dict:
        """提议修改标签树节点（始终需要用户裁决）。"""
        return await self._send({
            "type": "tag_tree",
            "entity_id": node_id,
            "entity_type": "tag_node",
            "payload": patch,
            "message": message,
        })

    async def propose_custom(
        self,
        type: str,
        entity_id: str,
        entity_type: str,
        payload: dict,
        message: str = "",
    ) -> dict:
        """通用提议接口，供自定义扩展。"""
        return await self._send({
            "type": type,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "payload": payload,
            "message": message,
        })


# 全局单例
sync = EgoneticsSync()
