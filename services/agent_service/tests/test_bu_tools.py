"""
Tests for ritecare_tools/tools/bu1-bu5 tools.

Each BU tool is an async function that calls an HTTP endpoint.
- BU1 uses resilient_request (patched via unittest.mock)
- BU2-BU5 use httpx.AsyncClient directly (patched via pytest-httpx)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── BU1 Tools (uses resilient_request) ──────────────────────────────

class TestBU1Tools:

    @pytest.mark.asyncio
    async def test_get_customer_by_id(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "C123", "name": "Alice"}

        with patch("ritecare_tools.tools.bu1_tools.resilient_request", new_callable=AsyncMock, return_value=mock_resp):
            from ritecare_tools.tools.bu1_tools import get_customer_by_id
            result = await get_customer_by_id("C123")
        assert result["id"] == "C123"
        assert result["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_onboarding_status(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"customer_id": "C123", "status": "completed", "kyc": True}

        with patch("ritecare_tools.tools.bu1_tools.resilient_request", new_callable=AsyncMock, return_value=mock_resp):
            from ritecare_tools.tools.bu1_tools import get_onboarding_status
            result = await get_onboarding_status("C123")
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_search_onboarding_docs(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"text": "Insurance doc chunk"}, {"text": "KYC requirements"}]}

        with patch("ritecare_tools.tools.bu1_tools.resilient_request", new_callable=AsyncMock, return_value=mock_resp):
            from ritecare_tools.tools.bu1_tools import search_onboarding_docs
            result = await search_onboarding_docs("insurance policy")
        assert result == ["Insurance doc chunk", "KYC requirements"]

    @pytest.mark.asyncio
    async def test_search_onboarding_docs_empty(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}

        with patch("ritecare_tools.tools.bu1_tools.resilient_request", new_callable=AsyncMock, return_value=mock_resp):
            from ritecare_tools.tools.bu1_tools import search_onboarding_docs
            result = await search_onboarding_docs("nonexistent topic")
        assert result == []


# ── BU2 Tools (uses httpx.AsyncClient directly) ────────────────────

class TestBU2Tools:

    @pytest.mark.asyncio
    async def test_get_contract_by_id_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/contracts/CON-456",
            json={"id": "CON-456", "type": "maintenance", "active": True},
        )
        from ritecare_tools.tools.bu2_tools import get_contract_by_id
        result = await get_contract_by_id("CON-456")
        assert result["id"] == "CON-456"

    @pytest.mark.asyncio
    async def test_get_contract_by_id_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/contracts/CON-999",
            status_code=404,
        )
        from ritecare_tools.tools.bu2_tools import get_contract_by_id
        result = await get_contract_by_id("CON-999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_contracts(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/contracts/customer/C100",
            json=[{"id": "CON-1"}, {"id": "CON-2"}],
        )
        from ritecare_tools.tools.bu2_tools import list_contracts
        result = await list_contracts("C100")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_contracts_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/contracts/customer/C999",
            status_code=404,
        )
        from ritecare_tools.tools.bu2_tools import list_contracts
        result = await list_contracts("C999")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_visits(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/visits/customer/C100",
            json=[{"id": "V1", "date": "2026-04-01"}],
        )
        from ritecare_tools.tools.bu2_tools import list_visits
        result = await list_visits("C100")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_search_service_manuals(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8002/rag/search",
            json={"results": [{"text": "HVAC maintenance step 1"}]},
        )
        from ritecare_tools.tools.bu2_tools import search_service_manuals
        result = await search_service_manuals("HVAC maintenance")
        assert result == ["HVAC maintenance step 1"]


# ── BU3 Tools ───────────────────────────────────────────────────────

class TestBU3Tools:

    @pytest.mark.asyncio
    async def test_get_subscription(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8003/subscriptions/C100",
            json={"customer_id": "C100", "plan": "premium"},
        )
        from ritecare_tools.tools.bu3_tools import get_subscription
        result = await get_subscription("C100")
        assert result["plan"] == "premium"

    @pytest.mark.asyncio
    async def test_get_subscription_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8003/subscriptions/C999",
            status_code=404,
        )
        from ritecare_tools.tools.bu3_tools import get_subscription
        result = await get_subscription("C999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_invoices(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8003/invoices/C100",
            json=[{"id": "INV-1", "amount": 150.00}],
        )
        from ritecare_tools.tools.bu3_tools import list_invoices
        result = await list_invoices("C100")
        assert len(result) == 1
        assert result[0]["amount"] == 150.00

    @pytest.mark.asyncio
    async def test_search_billing_statements(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8003/rag/search",
            json={"results": [{"text": "Payment terms: net 30"}]},
        )
        from ritecare_tools.tools.bu3_tools import search_billing_statements
        result = await search_billing_statements("payment terms")
        assert result == ["Payment terms: net 30"]


# ── BU4 Tools ───────────────────────────────────────────────────────

class TestBU4Tools:

    @pytest.mark.asyncio
    async def test_get_ticket_by_id(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8004/tickets/TK-101",
            json={"id": "TK-101", "status": "open", "priority": "high"},
        )
        from ritecare_tools.tools.bu4_tools import get_ticket_by_id
        result = await get_ticket_by_id("TK-101")
        assert result["status"] == "open"

    @pytest.mark.asyncio
    async def test_get_ticket_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8004/tickets/TK-999",
            status_code=404,
        )
        from ritecare_tools.tools.bu4_tools import get_ticket_by_id
        result = await get_ticket_by_id("TK-999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_tickets(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8004/tickets/customer/C100",
            json=[{"id": "TK-1"}, {"id": "TK-2"}],
        )
        from ritecare_tools.tools.bu4_tools import list_tickets
        result = await list_tickets("C100")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_search_knowledge_base(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8004/rag/search",
            json={"results": [{"text": "Password reset: go to settings"}]},
        )
        from ritecare_tools.tools.bu4_tools import search_knowledge_base
        result = await search_knowledge_base("password reset")
        assert "Password reset" in result[0]

    @pytest.mark.asyncio
    async def test_search_resolved_tickets(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8004/rag/search",
            json={"results": [{"text": "Resolved: fixed by restarting service"}]},
        )
        from ritecare_tools.tools.bu4_tools import search_resolved_tickets
        result = await search_resolved_tickets("service restart")
        assert len(result) == 1


# ── BU5 Tools ───────────────────────────────────────────────────────

class TestBU5Tools:

    def test_find_service_type_match(self):
        from ritecare_tools.tools.bu5_tools import _find_service_type
        assert _find_service_type("skilled-nursing procedures") == "skilled-nursing"
        assert _find_service_type("physical-therapy guidelines") == "physical-therapy"
        assert _find_service_type("occupational-therapy assessment") == "occupational-therapy"
        assert _find_service_type("respite-care schedule") == "respite-care"
        assert _find_service_type("personal-care-companionship duties") == "personal-care-companionship"

    def test_find_service_type_no_match(self):
        from ritecare_tools.tools.bu5_tools import _find_service_type
        assert _find_service_type("general question about visits") is None

    def test_find_service_type_case_insensitive(self):
        from ritecare_tools.tools.bu5_tools import _find_service_type
        assert _find_service_type("SKILLED-NURSING procedures") == "skilled-nursing"

    @pytest.mark.asyncio
    async def test_get_visit_by_id_success(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/visits/V789",
            json={"id": "V789", "patient": "Bob", "status": "scheduled"},
        )
        from ritecare_tools.tools.bu5_tools import get_visit_by_id
        result = await get_visit_by_id("V789")
        assert result["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_get_visit_by_id_not_found(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/visits/V999",
            status_code=404,
        )
        from ritecare_tools.tools.bu5_tools import get_visit_by_id
        result = await get_visit_by_id("V999")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_patient_visits(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/visits/patient/P100",
            json=[{"id": "V1"}, {"id": "V2"}],
        )
        from ritecare_tools.tools.bu5_tools import list_patient_visits
        result = await list_patient_visits("P100")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_patient_visits_empty(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/visits/patient/P_NONE",
            status_code=404,
        )
        from ritecare_tools.tools.bu5_tools import list_patient_visits
        result = await list_patient_visits("P_NONE")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_care_documents_with_service_type(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/rag/search",
            json={"results": [{"text": "Nursing protocol step 1"}]},
        )
        from ritecare_tools.tools.bu5_tools import search_care_documents
        result = await search_care_documents("skilled-nursing wound care")
        assert len(result) == 1
        assert "Nursing protocol" in result[0]

    @pytest.mark.asyncio
    async def test_search_care_documents_no_service_type(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/rag/search",
            json={"results": [{"text": "General care doc"}]},
        )
        from ritecare_tools.tools.bu5_tools import search_care_documents
        result = await search_care_documents("general care guidelines")
        assert result == ["General care doc"]

    @pytest.mark.asyncio
    async def test_search_care_documents_empty(self, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:8006/rag/search",
            status_code=404,
        )
        from ritecare_tools.tools.bu5_tools import search_care_documents
        result = await search_care_documents("nonexistent")
        assert result == []
