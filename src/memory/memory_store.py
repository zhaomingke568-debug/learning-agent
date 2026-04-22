import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = field(default=None)
    importance: float = 0.5  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    category: str = "general"  # "user_preference", "context", "task_state", "general"
    metadata: dict = field(default_factory=dict)

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1

    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    def idle_hours(self) -> float:
        return (time.time() - self.last_accessed) / 3600


class MemoryStore:
    """In-memory store with strategy-based cleanup."""

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_hours: float = 24 * 7,
    ):
        self._store: dict[str, MemoryEntry] = {}
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours

    def add(
        self,
        content: Any,
        importance: float = 0.5,
        category: str = "general",
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            importance=importance,
            category=category,
            metadata=metadata or {},
        )
        self._store[entry.id] = entry
        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        entry = self._store.get(entry_id)
        if entry:
            entry.touch()
        return entry

    def get_all(self) -> list[MemoryEntry]:
        return list(self._store.values())

    def update_importance(self, entry_id: str, importance: float):
        if entry := self._store.get(entry_id):
            entry.importance = max(0.0, min(1.0, importance))

    def delete(self, entry_id: str) -> bool:
        return self._store.pop(entry_id, None) is not None

    def filter(self, category: Optional[str] = None) -> list[MemoryEntry]:
        entries = self.get_all()
        if category:
            entries = [e for e in entries if e.category == category]
        return entries

    def clear(self, category: Optional[str] = None):
        if category:
            to_remove = [e.id for e in self.filter(category)]
            for rid in to_remove:
                self._store.pop(rid, None)
        else:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def is_near_capacity(self, threshold: float = 0.8) -> bool:
        return len(self) >= self.max_entries * threshold
