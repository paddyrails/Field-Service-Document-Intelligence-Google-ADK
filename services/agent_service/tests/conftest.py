"""
Shared fixtures used across all test modules.

Key patterns:
- env vars set at module level so Settings() works on first import
- fake_mongo_collection: an in-memory dict that quacks like a Motor collection
"""
import os

# ── Must be set BEFORE any application module is imported ───────────
# Settings() is called at module scope in shared/config.py, so env vars
# must exist before Python processes those imports.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB_NAME", "test_ritecare")

import pytest
from unittest.mock import MagicMock


# ── Fake Motor collection (in-memory) ──────────────────────────────
class FakeCollection:
    """Dict-backed async replacement for a Motor collection."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def insert_one(self, doc):
        key = doc.get("id") or str(len(self._store))
        self._store[key] = doc
        return MagicMock(inserted_id=key)

    async def find_one(self, query):
        session_id = query.get("id")
        doc = self._store.get(session_id)
        if doc:
            return {**doc, "_id": "mongo_oid"}  # simulate Mongo adding _id
        return None

    async def delete_one(self, query):
        self._store.pop(query.get("id"), None)

    async def update_one(self, query, update):
        doc = self._store.get(query.get("id"))
        if not doc:
            return
        if "$push" in update:
            for field, value in update["$push"].items():
                doc.setdefault(field, []).append(value)
        if "$set" in update:
            for field, value in update["$set"].items():
                doc[field] = value

    def find(self, query):
        """Return an async iterator over matching docs."""
        matches = [
            {**doc, "_id": "mongo_oid"}
            for doc in self._store.values()
            if all(doc.get(k) == v for k, v in query.items())
        ]
        return _AsyncIterator(matches)


class _AsyncIterator:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


@pytest.fixture
def fake_collection():
    return FakeCollection()
