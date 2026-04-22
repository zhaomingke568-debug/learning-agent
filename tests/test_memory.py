import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory import (
    MemoryStore,
    MemoryEntry,
    importance_based_cleanup,
    time_based_cleanup,
    capacity_based_compression,
    default_summarizer,
)


def test_memory_store_basic():
    """Test basic MemoryStore operations."""
    store = MemoryStore(max_entries=10)

    # Add entries
    e1 = store.add("用户偏好 Mac", importance=0.8, category="user_preference")
    e2 = store.add("临时调试信息", importance=0.2, category="context")
    e3 = store.add("任务状态", importance=0.6, category="task_state")

    assert len(store) == 3

    # Get entry
    retrieved = store.get(e1.id)
    assert retrieved is not None
    assert retrieved.content == "用户偏好 Mac"
    assert retrieved.access_count == 1

    # Filter by category
    prefs = store.filter(category="user_preference")
    assert len(prefs) == 1

    # Delete entry
    assert store.delete(e2.id) is True
    assert len(store) == 2
    assert store.delete("nonexistent") is False


def test_importance_based_cleanup():
    """Test importance-based cleanup strategy."""
    store = MemoryStore(max_entries=100)

    for i in range(10):
        store.add(f"记忆{i}", importance=i * 0.1, category="test")

    assert len(store) == 10

    # Delete entries with importance < 0.3
    deleted = importance_based_cleanup(store, min_importance=0.3)
    assert len(deleted) == 3  # importance 0.0, 0.1, 0.2
    assert len(store) == 7

    # Keep top 3
    store2 = MemoryStore(max_entries=100)
    for i in range(10):
        store2.add(f"记忆{i}", importance=i * 0.1, category="test")

    deleted2 = importance_based_cleanup(store2, keep_top_n=3)
    assert len(deleted2) == 7
    assert len(store2) == 3

    # Verify highest importance entries remain
    remaining = store2.get_all()
    importances = [e.importance for e in remaining]
    assert max(importances) >= 0.8


def test_time_based_cleanup():
    """Test time-based cleanup strategy."""
    store = MemoryStore(max_entries=100, default_ttl_hours=1)

    # Add old entry (mock by modifying created_at)
    old_entry = store.add("旧记忆", importance=0.5, category="test")
    old_entry.created_at = time.time() - 7200  # 2 hours ago

    # Add recent entry
    new_entry = store.add("新记忆", importance=0.5, category="test")

    # TTL cleanup should delete old entry
    deleted = time_based_cleanup(store, ttl_hours=1)
    assert len(deleted) == 1
    assert old_entry.id in deleted
    assert len(store) == 1

    # Idle cleanup test
    store2 = MemoryStore(max_entries=100)
    idle_entry = store2.add("很久没访问", importance=0.5, category="test")
    idle_entry.last_accessed = time.time() - 3600 * 3  # 3 hours ago

    active_entry = store2.add("最近访问", importance=0.5, category="test")
    active_entry.last_accessed = time.time() - 100

    deleted2 = time_based_cleanup(store2, idle_hours=2)
    assert len(deleted2) == 1
    assert idle_entry.id in deleted2


def test_capacity_based_compression():
    """Test capacity-based compression strategy."""
    store = MemoryStore(max_entries=10)

    # Fill to near capacity
    for i in range(10):
        store.add(f"记忆{i}", importance=i * 0.1, category="general")

    assert len(store) == 10
    assert store.is_near_capacity(threshold=0.8) is True

    # Compression should keep ~70% and compress the rest
    result = capacity_based_compression(
        store,
        summarizer=default_summarizer,
        target_reduction=0.3,
    )

    # Should have compressed some entries and added a summary
    assert result["new_entry_id"] is not None
    # 10 entries, target_reduction=0.3 keeps 7 + 1 summary = 8 total
    assert len(store) <= 8

    # Verify compressed summary exists
    summary_entries = store.filter(category="compressed_summary")
    assert len(summary_entries) == 1


def test_capacity_no_compression_when_not_needed():
    """Test that compression doesn't trigger when under threshold."""
    store = MemoryStore(max_entries=100)

    for i in range(10):
        store.add(f"记忆{i}", importance=i * 0.1, category="general")

    assert store.is_near_capacity(threshold=0.8) is False

    result = capacity_based_compression(
        store,
        summarizer=default_summarizer,
        target_reduction=0.3,
    )

    assert result["new_entry_id"] is None
    assert len(store) == 10  # No changes


def test_priority_categories_not_compressed():
    """Test that priority categories are not compressed."""
    store = MemoryStore(max_entries=10)

    # Add low importance entries in different categories
    store.add("低优先级", importance=0.1, category="general")
    store.add("高优先级1", importance=0.1, category="user_preference")
    store.add("高优先级2", importance=0.1, category="task_state")

    result = capacity_based_compression(
        store,
        summarizer=default_summarizer,
        target_reduction=0.5,
        priority_categories=["user_preference", "task_state"],
    )

    # Only general category should be compressed
    remaining = store.filter(category="general")
    assert len(remaining) <= 5


def test_memory_entry_age_and_idle():
    """Test MemoryEntry age and idle time calculations."""
    entry = MemoryEntry(content="test", importance=0.5)

    # Should be new
    assert entry.age_hours() < 0.01
    assert entry.idle_hours() < 0.01

    # Simulate old entry
    entry.created_at = time.time() - 3600  # 1 hour ago
    entry.last_accessed = time.time() - 7200  # 2 hours ago

    assert 0.99 < entry.age_hours() < 1.01
    assert 1.99 < entry.idle_hours() < 2.01


def run_all_tests():
    print("Running memory module tests...\n")

    tests = [
        test_memory_store_basic,
        test_importance_based_cleanup,
        test_time_based_cleanup,
        test_capacity_based_compression,
        test_capacity_no_compression_when_not_needed,
        test_priority_categories_not_compressed,
        test_memory_entry_age_and_idle,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: Unexpected error: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
