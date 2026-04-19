"""
Tests for agent/session.py — MongoSessionService.

Uses FakeCollection from conftest so no real Mongo needed.
Covers: create, get (existing + auto-create), list, delete, append_event, _sanitize.
"""
import pytest
from unittest.mock import MagicMock


class TestSanitize:
    """_sanitize converts sets to lists for BSON compatibility."""

    def test_set_to_list(self):
        from agent.session import _sanitize
        result = _sanitize({"tags": {"a", "b"}})
        assert isinstance(result["tags"], list)
        assert set(result["tags"]) == {"a", "b"}

    def test_nested_set(self):
        from agent.session import _sanitize
        result = _sanitize({"outer": {"inner": {1, 2}}})
        assert isinstance(result["outer"]["inner"], list)

    def test_list_with_sets(self):
        from agent.session import _sanitize
        result = _sanitize([{1, 2}, {3, 4}])
        assert all(isinstance(item, list) for item in result)

    def test_plain_values_unchanged(self):
        from agent.session import _sanitize
        assert _sanitize("hello") == "hello"
        assert _sanitize(42) == 42
        assert _sanitize(None) is None

    def test_deeply_nested(self):
        from agent.session import _sanitize
        result = _sanitize({"a": {"b": {"c": {"d", "e"}}}})
        assert isinstance(result["a"]["b"]["c"], list)


class TestMongoSessionService:
    """Integration tests for MongoSessionService with in-memory collection."""

    @pytest.fixture
    def service(self, fake_collection):
        from agent.session import MongoSessionService
        svc = MongoSessionService.__new__(MongoSessionService)
        svc.db = MagicMock()
        svc.collection = fake_collection
        return svc

    @pytest.mark.asyncio
    async def test_create_session(self, service):
        session = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="S1", state={"role": "admin"}
        )
        assert session.app_name == "ritecare"
        assert session.user_id == "U1"
        assert session.state["role"] == "admin"

    @pytest.mark.asyncio
    async def test_create_session_with_id(self, service):
        session = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="custom-id"
        )
        assert session.id == "custom-id"

    @pytest.mark.asyncio
    async def test_create_session_default_state(self, service):
        session = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        assert session.state == {}

    @pytest.mark.asyncio
    async def test_get_existing_session(self, service):
        created = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        fetched = await service.get_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_missing_session_auto_creates(self, service):
        """get_session auto-creates if session_id doesn't exist."""
        session = await service.get_session(
            app_name="ritecare", user_id="U1", session_id="new-session"
        )
        assert session is not None
        assert session.id == "new-session"

    @pytest.mark.asyncio
    async def test_list_sessions(self, service):
        await service.create_session(app_name="ritecare", user_id="U1", session_id="S1")
        await service.create_session(app_name="ritecare", user_id="U1", session_id="S2")
        await service.create_session(app_name="ritecare", user_id="U2", session_id="S3")

        sessions = await service.list_sessions(app_name="ritecare", user_id="U1")
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, service):
        sessions = await service.list_sessions(app_name="ritecare", user_id="NOBODY")
        assert sessions == []

    @pytest.mark.asyncio
    async def test_delete_session(self, service):
        await service.create_session(app_name="ritecare", user_id="U1", session_id="S1")
        await service.delete_session(app_name="ritecare", user_id="U1", session_id="S1")
        # After delete, get_session should auto-create a fresh one
        session = await service.get_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        assert session.state == {}  # fresh session, empty state

    @pytest.mark.asyncio
    async def test_append_event(self, service):
        session = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        event = MagicMock()
        event.model_dump.return_value = {"type": "user_message", "text": "hello"}
        session.events = []

        await service.append_event(session, event)
        assert len(session.events) == 1

    @pytest.mark.asyncio
    async def test_append_multiple_events(self, service):
        session = await service.create_session(
            app_name="ritecare", user_id="U1", session_id="S1"
        )
        session.events = []

        for i in range(3):
            event = MagicMock()
            event.model_dump.return_value = {"type": "message", "text": f"msg-{i}"}
            await service.append_event(session, event)

        assert len(session.events) == 3
