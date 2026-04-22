import time
from typing import Callable, Protocol

from .memory_store import MemoryEntry, MemoryStore


class Summarizer(Protocol):
    """Protocol for memory summarization functions."""

    def __call__(self, entries: list[MemoryEntry]) -> str:
        ...


# --- Importance-Based Cleanup ---


def importance_based_cleanup(
    store: MemoryStore,
    keep_top_n: int | None = None,
    min_importance: float = 0.3,
    category: str | None = None,
) -> list[str]:
    """
    删除低重要性记忆。

    Args:
        store: MemoryStore 实例
        keep_top_n: 至少保留 N 条最重要的记忆（忽略 min_importance）
        min_importance: 删除低于此重要性分数的记忆
        category: 仅清理指定分类的记忆

    Returns:
        被删除的记忆 ID 列表
    """
    entries = store.filter(category=category)
    deleted_ids = []

    if keep_top_n is not None and len(entries) > keep_top_n:
        sorted_entries = sorted(entries, key=lambda e: e.importance, reverse=True)
        to_keep = set(e.id for e in sorted_entries[:keep_top_n])
        to_delete = [e for e in entries if e.id not in to_keep]
    else:
        to_delete = [e for e in entries if e.importance < min_importance]

    for entry in to_delete:
        if store.delete(entry.id):
            deleted_ids.append(entry.id)

    return deleted_ids


# --- Time-Based Cleanup ---


def time_based_cleanup(
    store: MemoryStore,
    ttl_hours: float | None = None,
    idle_hours: float | None = None,
    category: str | None = None,
) -> list[str]:
    """
    删除过期或长期未访问的记忆。

    Args:
        store: MemoryStore 实例
        ttl_hours: 记忆创建后超过此时间则删除（None 则使用 store.default_ttl_hours）
        idle_hours: 最后访问时间超过此时间则删除
        category: 仅清理指定分类的记忆

    Returns:
        被删除的记忆 ID 列表
    """
    ttl = ttl_hours if ttl_hours is not None else store.default_ttl_hours
    entries = store.filter(category=category)
    deleted_ids = []

    for entry in entries:
        should_delete = False

        if ttl > 0 and entry.age_hours() > ttl:
            should_delete = True
        elif idle_hours is not None and entry.idle_hours() > idle_hours:
            should_delete = True

        if should_delete and store.delete(entry.id):
            deleted_ids.append(entry.id)

    return deleted_ids


# --- Capacity-Based Compression ---


def capacity_based_compression(
    store: MemoryStore,
    summarizer: Summarizer,
    target_reduction: float = 0.3,
    priority_categories: list[str] | None = None,
) -> dict:
    """
    当存储接近上限时，对低重要性记忆进行摘要压缩。

    Args:
        store: MemoryStore 实例
        summarizer: 接收记忆列表，返回摘要字符串的函数
        target_reduction: 目标压缩比例（0.0 - 1.0）
        priority_categories: 高优先级分类列表，这些分类的记忆不会被压缩

    Returns:
        {"summarized": ..., "deleted": ..., "new_entry_id": ...}
    """
    if not store.is_near_capacity():
        return {"summarized": [], "deleted": [], "new_entry_id": None}

    priority_categories = priority_categories or []
    threshold = store.max_entries * (1 - target_reduction)

    entries = [
        e for e in store.get_all()
        if e.category not in priority_categories
    ]
    entries.sort(key=lambda e: e.importance)

    keep_count = max(1, int(threshold))
    to_compress = entries[: max(0, len(entries) - keep_count)]

    if not to_compress:
        return {"summarized": [], "deleted": [], "new_entry_id": None}

    deleted_ids = []
    for entry in to_compress:
        if store.delete(entry.id):
            deleted_ids.append(entry.id)

    summary_content = summarizer(to_compress)
    new_entry = store.add(
        content=summary_content,
        importance=0.6,
        category="compressed_summary",
        metadata={"compressed_count": len(to_compress)},
    )

    return {
        "summarized": [e.id for e in to_compress],
        "deleted": deleted_ids,
        "new_entry_id": new_entry.id,
    }


# --- Default LLM Summarizer (placeholder) ---


def default_summarizer(entries: list[MemoryEntry]) -> str:
    """默认摘要器：将记忆列表合并为单条摘要文本。"""
    if not entries:
        return ""
    parts = []
    for e in entries:
        parts.append(f"[{e.category}] {e.content}")
    return " | ".join(parts)
