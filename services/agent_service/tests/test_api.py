"""
Tests for agent_api/main.py — the FastAPI app.

Uses httpx.AsyncClient with ASGITransport (no real server needed).
The /query endpoint is heavily mocked since it triggers the full ADK Runner.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def test_client():
    from agent_api.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, test_client):
        async with test_client as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestQueryEndpoint:

    @pytest.mark.asyncio
    async def test_query_returns_response(self, test_client):
        """Mock runner.run_async to avoid hitting Gemini."""

        fake_event = MagicMock()
        fake_event.is_final_response.return_value = True
        fake_event.content = MagicMock()
        fake_event.content.parts = [MagicMock(text="Visit V123 is scheduled for Monday")]

        async def fake_run_async(**kwargs):
            yield fake_event

        with patch("agent_api.main.runner") as mock_runner:
            mock_runner.run_async = fake_run_async
            async with test_client as client:
                resp = await client.post("/query", json={
                    "query": "What is the status of visit V123?",
                    "user_id": "U1",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "V123" in data["response"]
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_query_empty_response(self, test_client):
        """If no final event, response is empty string."""

        async def fake_run_async(**kwargs):
            event = MagicMock()
            event.is_final_response.return_value = False
            yield event

        with patch("agent_api.main.runner") as mock_runner:
            mock_runner.run_async = fake_run_async
            async with test_client as client:
                resp = await client.post("/query", json={
                    "query": "patient visit status",
                    "user_id": "U1",
                })

        assert resp.status_code == 200
        assert resp.json()["response"] == ""

    @pytest.mark.asyncio
    async def test_query_validates_body(self, test_client):
        """Missing required 'query' field returns 422."""
        async with test_client as client:
            resp = await client.post("/query", json={"user_id": "U1"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_default_fields(self, test_client):
        """session_id, channel, user_id all have defaults."""

        fake_event = MagicMock()
        fake_event.is_final_response.return_value = True
        fake_event.content = MagicMock()
        fake_event.content.parts = [MagicMock(text="OK")]

        async def fake_run_async(**kwargs):
            yield fake_event

        with patch("agent_api.main.runner") as mock_runner:
            mock_runner.run_async = fake_run_async
            async with test_client as client:
                resp = await client.post("/query", json={"query": "patient visit"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"]  # should have auto-generated UUID
        assert data["response"] == "OK"

    @pytest.mark.asyncio
    async def test_query_preserves_session_id(self, test_client):
        """Custom session_id is preserved in response."""

        fake_event = MagicMock()
        fake_event.is_final_response.return_value = True
        fake_event.content = MagicMock()
        fake_event.content.parts = [MagicMock(text="OK")]

        async def fake_run_async(**kwargs):
            yield fake_event

        with patch("agent_api.main.runner") as mock_runner:
            mock_runner.run_async = fake_run_async
            async with test_client as client:
                resp = await client.post("/query", json={
                    "query": "patient visit",
                    "session_id": "my-custom-session",
                })

        assert resp.json()["session_id"] == "my-custom-session"
