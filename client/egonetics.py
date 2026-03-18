"""
Egonetics API client.
Reads/writes Task, Graph (canvas nodes/relations), Pages from Egonetics backend.
"""
import httpx
from typing import Optional, Any
from config.settings import settings


class EgoneticsClient:
    def __init__(self):
        self.base = settings.egonetics_url
        self._token = settings.egonetics_token

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _get(self, path: str, params: dict = None) -> Any:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{self.base}{path}", headers=self.headers, params=params)
            r.raise_for_status()
            return r.json()

    async def _post(self, path: str, body: dict = None) -> Any:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self.base}{path}", headers=self.headers, json=body or {})
            r.raise_for_status()
            return r.json()

    async def _patch(self, path: str, body: dict) -> Any:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.patch(f"{self.base}{path}", headers=self.headers, json=body)
            r.raise_for_status()
            return r.json()

    async def _delete(self, path: str) -> Any:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.delete(f"{self.base}{path}", headers=self.headers)
            r.raise_for_status()
            return r.json()

    # -- Tasks --------------------------------------------------------------
    async def get_task(self, task_id: str) -> dict:
        return await self._get(f"/api/tasks/{task_id}")

    async def list_tasks(self, column_id: str = None) -> list:
        params = {"column_id": column_id} if column_id else None
        return await self._get("/api/tasks", params)

    async def update_task(self, task_id: str, patch: dict) -> dict:
        return await self._patch(f"/api/tasks/{task_id}", patch)

    async def get_task_pages(self, task_id: str) -> list:
        return await self._get(f"/api/tasks/{task_id}/pages")

    # -- Canvas / Graph -----------------------------------------------------
    async def list_canvases(self) -> list:
        return await self._get("/api/canvases")

    async def get_canvas(self, canvas_id: str) -> dict:
        return await self._get(f"/api/canvases/{canvas_id}")

    async def create_canvas(self, title: str, description: str = "") -> dict:
        return await self._post("/api/canvases", {"title": title, "description": description})

    async def get_nodes(self, canvas_id: str) -> list:
        return await self._get(f"/api/canvases/{canvas_id}/nodes")

    async def add_node(self, canvas_id: str, entity_type: str, entity_id: str,
                       x: float = 100, y: float = 100,
                       node_kind: str = "entity", exec_config: dict = None) -> dict:
        return await self._post(f"/api/canvases/{canvas_id}/nodes", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "x": x, "y": y,
            "node_kind": node_kind,
            "exec_config": exec_config or {}
        })

    async def update_node(self, canvas_id: str, node_id: str, patch: dict) -> dict:
        return await self._patch(f"/api/canvases/{canvas_id}/nodes/{node_id}", patch)

    async def set_node_lifecycle(self, canvas_id: str, node_id: str,
                                  state: str, cost_snapshot: dict = None) -> dict:
        patch = {"lifecycle_state": state}
        if cost_snapshot:
            patch["cost_snapshot"] = cost_snapshot
        return await self.update_node(canvas_id, node_id, patch)

    async def delete_node(self, canvas_id: str, node_id: str) -> dict:
        return await self._delete(f"/api/canvases/{canvas_id}/nodes/{node_id}")

    # -- Relations ---------------------------------------------------------
    async def get_relations(self, source_id: str = None, target_id: str = None) -> list:
        params = {}
        if source_id: params["source_id"] = source_id
        if target_id: params["target_id"] = target_id
        return await self._get("/api/relations", params)

    async def create_relation(self, source_type: str, source_id: str,
                               target_type: str, target_id: str,
                               relation_type: str = "contains", title: str = "") -> dict:
        return await self._post("/api/relations", {
            "source_type": source_type, "source_id": source_id,
            "target_type": target_type, "target_id": target_id,
            "relation_type": relation_type, "title": title
        })

    # -- Pages / Knowledge -------------------------------------------------
    async def list_pages(self, root_only: bool = True, page_type: str = None) -> list:
        params = {"rootOnly": "true" if root_only else "false"}
        if page_type: params["type"] = page_type
        return await self._get("/api/pages", params)

    async def get_page(self, page_id: str) -> dict:
        return await self._get(f"/api/pages/{page_id}")

    async def get_page_blocks(self, page_id: str) -> list:
        return await self._get(f"/api/pages/{page_id}/blocks")

    async def search_pages(self, query: str) -> list:
        return await self._get("/api/pages", {"search": query})

    # -- Relation types ----------------------------------------------------
    async def get_relation_types(self) -> list:
        return await self._get("/api/relation-types")

    # -- Execution Canvas --------------------------------------------------

    async def create_execution_canvas(self, task_id: str, title: str) -> dict:
        """Create a canvas of type 'execution' linked to a task."""
        return await self._post("/api/canvases", {
            "title": title,
            "canvasType": "execution",
            "taskRefId": task_id,
        })

    async def get_execution_canvases(self, task_id: str = None) -> list:
        """List execution canvases, optionally filtered by task_id."""
        canvases = await self._get("/api/canvases", {"type": "execution"})
        if task_id:
            canvases = [c for c in canvases if c.get("task_ref_id") == task_id]
        return canvases

    # -- Exec Step Pages ---------------------------------------------------

    async def create_exec_step_page(self, task_id: str, title: str, icon: str = "⚙️") -> dict:
        """Create an independent exec_step page linked to a task via ref_id."""
        return await self._post("/api/pages", {
            "title": title,
            "icon": icon,
            "pageType": "exec_step",
            "refId": task_id,
        })

    async def get_exec_steps(self, task_id: str) -> list:
        """Get all exec_step pages for a task, ordered by created_at ASC."""
        return await self._get("/api/pages", {"taskRefId": task_id})

    async def append_block_to_page(self, page_id: str, block_type: str,
                                    content: dict, creator: str = "agent") -> dict:
        """Append a block to a page (exec_step output, etc.)."""
        return await self._post(f"/api/pages/{page_id}/blocks", {
            "type": block_type,
            "content": content,
            "creator": creator,
        })


egonetics = EgoneticsClient()
