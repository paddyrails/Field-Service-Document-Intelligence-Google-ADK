"""
Tests for shared/http_client.py — resilient_request with tenacity retries.

Uses pytest-httpx to intercept outgoing httpx calls.
"""
import pytest
import httpx
from tenacity import RetryError


class TestResilientRequest:

    @pytest.mark.asyncio
    async def test_success_on_first_try(self, httpx_mock):
        httpx_mock.add_response(url="http://bu1:8001/test", json={"ok": True})
        from shared.http_client import resilient_request
        resp = await resilient_request("GET", "http://bu1:8001/test", "bu1")
        assert resp.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self, httpx_mock):
        """Should retry up to 3 times on ConnectError, then succeed."""
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_response(url="http://bu1:8001/test", json={"ok": True})

        from shared.http_client import resilient_request
        resp = await resilient_request("GET", "http://bu1:8001/test", "bu1")
        assert resp.json() == {"ok": True}

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, httpx_mock):
        """After 3 failed attempts, the exception propagates."""
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_exception(httpx.ConnectError("refused"))

        from shared.http_client import resilient_request
        with pytest.raises(RetryError):
            await resilient_request("GET", "http://bu1:8001/test", "bu1")

    @pytest.mark.asyncio
    async def test_http_error_raised_immediately(self, httpx_mock):
        """Non-retryable HTTP errors (e.g. 500) raise immediately."""
        httpx_mock.add_response(url="http://bu1:8001/test", status_code=500)
        from shared.http_client import resilient_request
        with pytest.raises(httpx.HTTPStatusError):
            await resilient_request("GET", "http://bu1:8001/test", "bu1")

    @pytest.mark.asyncio
    async def test_post_with_json_body(self, httpx_mock):
        httpx_mock.add_response(
            url="http://bu1:8001/rag/search",
            json={"results": [{"text": "doc1"}]},
        )
        from shared.http_client import resilient_request
        resp = await resilient_request(
            "POST", "http://bu1:8001/rag/search", "bu1",
            json={"query": "test", "top_k": 5},
        )
        assert resp.json()["results"][0]["text"] == "doc1"

    @pytest.mark.asyncio
    async def test_retries_on_read_timeout(self, httpx_mock):
        """ReadTimeout is also in the retry filter."""
        httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
        httpx_mock.add_response(url="http://bu1:8001/test", json={"ok": True})

        from shared.http_client import resilient_request
        resp = await resilient_request("GET", "http://bu1:8001/test", "bu1")
        assert resp.json() == {"ok": True}
