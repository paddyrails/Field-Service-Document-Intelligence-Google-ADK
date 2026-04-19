"""
Tests for ritecare_tools/tools/rag_tools.py

Covers: _rerank, search_bu_documents, search_all_bus.
Mocks: Gemini embeddings, MongoDB aggregation, CrossEncoder.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRerank:

    def test_rerank_sorts_by_score(self):
        from ritecare_tools.tools.rag_tools import _reranker

        with patch.object(_reranker, "predict", return_value=[0.1, 0.9, 0.5]):
            from ritecare_tools.tools.rag_tools import _rerank
            chunks = [
                {"text": "low"},
                {"text": "high"},
                {"text": "mid"},
            ]
            result = _rerank("query", chunks, top_k=2)
        assert result[0]["text"] == "high"
        assert result[1]["text"] == "mid"
        assert len(result) == 2

    def test_rerank_empty_input(self):
        from ritecare_tools.tools.rag_tools import _rerank
        assert _rerank("query", [], top_k=5) == []

    def test_rerank_single_item(self):
        from ritecare_tools.tools.rag_tools import _reranker

        with patch.object(_reranker, "predict", return_value=[0.8]):
            from ritecare_tools.tools.rag_tools import _rerank
            chunks = [{"text": "only one"}]
            result = _rerank("query", chunks, top_k=5)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.8

    def test_rerank_top_k_limits_output(self):
        from ritecare_tools.tools.rag_tools import _reranker

        with patch.object(_reranker, "predict", return_value=[0.3, 0.7, 0.5, 0.9, 0.1]):
            from ritecare_tools.tools.rag_tools import _rerank
            chunks = [{"text": f"doc-{i}"} for i in range(5)]
            result = _rerank("query", chunks, top_k=3)
        assert len(result) == 3
        # Should be sorted descending by rerank_score
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]
        assert result[1]["rerank_score"] >= result[2]["rerank_score"]


class TestSearchBuDocuments:

    @pytest.mark.asyncio
    async def test_unknown_bu_returns_error(self):
        from ritecare_tools.tools.rag_tools import search_bu_documents
        result = await search_bu_documents("test query", bu="BU99")
        assert result[0].get("error")
        assert "Unknown BU" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_valid_bu_search(self):
        fake_embed_response = MagicMock()
        fake_embed_response.embeddings = [MagicMock(values=[0.1] * 3072)]

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_embed_response)

        # Fake Mongo aggregation result
        async def fake_aggregate(pipeline):
            for doc in [{"text": "chunk1", "metadata": {}, "score": 0.95}]:
                yield doc

        fake_collection = MagicMock()
        fake_collection.aggregate = fake_aggregate
        fake_db = MagicMock()
        fake_db.__getitem__ = MagicMock(return_value=fake_collection)

        with patch("ritecare_tools.tools.rag_tools._genai_client", mock_client), \
             patch("ritecare_tools.tools.rag_tools.get_database", return_value=fake_db), \
             patch("ritecare_tools.tools.rag_tools._reranker") as mock_reranker:
            mock_reranker.predict.return_value = [0.95]
            from ritecare_tools.tools.rag_tools import search_bu_documents
            result = await search_bu_documents("nursing protocol", bu="BU5", top_k=3)

        assert len(result) >= 1
        assert result[0]["text"] == "chunk1"
        assert result[0]["bu"] == "BU5"

    @pytest.mark.asyncio
    async def test_valid_bu_search_empty_results(self):
        fake_embed_response = MagicMock()
        fake_embed_response.embeddings = [MagicMock(values=[0.1] * 3072)]

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_embed_response)

        async def fake_aggregate(pipeline):
            return
            yield  # make it an async generator that yields nothing

        fake_collection = MagicMock()
        fake_collection.aggregate = fake_aggregate
        fake_db = MagicMock()
        fake_db.__getitem__ = MagicMock(return_value=fake_collection)

        with patch("ritecare_tools.tools.rag_tools._genai_client", mock_client), \
             patch("ritecare_tools.tools.rag_tools.get_database", return_value=fake_db), \
             patch("ritecare_tools.tools.rag_tools._reranker") as mock_reranker:
            mock_reranker.predict.return_value = []
            from ritecare_tools.tools.rag_tools import search_bu_documents
            result = await search_bu_documents("nonexistent", bu="BU1", top_k=3)

        assert result == []


class TestSearchAllBus:

    @pytest.mark.asyncio
    async def test_search_all_bus_merges_results(self):
        fake_embed_response = MagicMock()
        fake_embed_response.embeddings = [MagicMock(values=[0.1] * 3072)]

        mock_client = MagicMock()
        mock_client.aio.models.embed_content = AsyncMock(return_value=fake_embed_response)

        # Each BU returns one chunk
        async def fake_aggregate(pipeline):
            yield {"text": "bu chunk", "metadata": {}, "score": 0.8}

        fake_collection = MagicMock()
        fake_collection.aggregate = fake_aggregate
        fake_db = MagicMock()
        fake_db.__getitem__ = MagicMock(return_value=fake_collection)

        with patch("ritecare_tools.tools.rag_tools._genai_client", mock_client), \
             patch("ritecare_tools.tools.rag_tools.get_database", return_value=fake_db), \
             patch("ritecare_tools.tools.rag_tools._reranker") as mock_reranker:
            # 5 BUs × 1 chunk each = 5 chunks
            mock_reranker.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
            from ritecare_tools.tools.rag_tools import search_all_bus
            result = await search_all_bus("cross-BU query", top_k=3)

        assert len(result) == 3
        # Should be sorted by rerank_score descending
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]
