"""WebSocket connection manager. Singleton used by lifecycle routes and agent loop."""
import logging
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        # task_id → list of active WebSockets
        self._sockets: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, ws: WebSocket):
        await ws.accept()
        self._sockets.setdefault(task_id, []).append(ws)
        logger.info(f"WS connected: task={task_id}, total={len(self._sockets[task_id])}")

    def disconnect(self, task_id: str, ws: WebSocket):
        bucket = self._sockets.get(task_id, [])
        if ws in bucket:
            bucket.remove(ws)
        if not bucket:
            self._sockets.pop(task_id, None)

    async def emit(self, task_id: str, event_type: str, data: dict = None):
        """Broadcast event to all clients watching task_id."""
        payload = {
            "type": event_type,
            "task_id": task_id,
            "ts": datetime.now().isoformat(),
            **(data or {}),
        }
        dead = []
        for ws in self._sockets.get(task_id, []):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(task_id, ws)

    def connected_tasks(self) -> list[str]:
        return list(self._sockets.keys())


ws_manager = ConnectionManager()
