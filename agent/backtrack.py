"""Backtracking state manager for the agent loop."""
from dataclasses import dataclass, field
from typing import Optional
from config.settings import settings


@dataclass
class BacktrackState:
    path: list[str] = field(default_factory=list)
    depth: int = 0
    attempts: dict[str, int] = field(default_factory=dict)  # node_id → attempt count


class BacktrackManager:

    def __init__(self):
        self._states: dict[str, BacktrackState] = {}  # task_id → state

    def get_state(self, task_id: str) -> BacktrackState:
        if task_id not in self._states:
            self._states[task_id] = BacktrackState()
        return self._states[task_id]

    def record_visit(self, task_id: str, node_id: str) -> int:
        state = self.get_state(task_id)
        state.path.append(node_id)
        state.depth = len(state.path)
        state.attempts[node_id] = state.attempts.get(node_id, 0) + 1
        return state.attempts[node_id]

    def can_backtrack(self, task_id: str) -> bool:
        state = self.get_state(task_id)
        return state.depth > 0 and state.depth <= settings.max_backtrack_depth

    def backtrack(self, task_id: str) -> Optional[str]:
        state = self.get_state(task_id)
        if state.path:
            return state.path.pop()
        return None

    def is_looping(self, task_id: str, node_id: str) -> bool:
        state = self.get_state(task_id)
        return state.attempts.get(node_id, 0) >= settings.max_loop_detection_count

    def reset(self, task_id: str):
        self._states.pop(task_id, None)
