"""
Agent Evaluator tests using Google ADK's built-in AgentEvaluator.

These are INTEGRATION tests — they call the real Gemini model.
Run them separately from unit tests:
    pytest tests/eval/ -v -m "eval" --timeout=120

What's tested:
- Tool trajectory: does the agent pick the correct tools for each query?
- Response quality: does the agent's response match the reference?
- RBAC: does the agent respect role-based tool access?
"""
import pytest
from google.adk.evaluation import AgentEvaluator


# ── Mark all tests in this module as eval (slow, needs API key) ─────
pytestmark = [
    pytest.mark.eval,
    pytest.mark.asyncio,
]


class TestToolRouting:
    """
    Validates the agent routes queries to the correct BU tools.

    Uses the OLD format (.test.json = list of dicts with query/expected_tool_use/reference).
    AgentEvaluator.evaluate() loads the file, runs the agent N times,
    computes tool_trajectory_avg_score and response_match_score,
    and asserts they meet thresholds from test_config.json.
    """

    async def test_tool_trajectory_all_bus(self):
        """Each BU query should trigger the expected tool(s)."""
        await AgentEvaluator.evaluate(
            agent_module="agent.agent",
            eval_dataset_file_path_or_dir="tests/eval/ritecare_tool_routing.test.json",
            num_runs=2,
            initial_session_file="tests/eval/initial_session_admin.json",
        )


class TestResponseQuality:
    """
    Validates response quality using the new EvalSet schema.

    Uses AgentEvaluator.evaluate() with the EvalSet JSON that contains
    Invocation objects with session_input (so RBAC state is injected).
    """

    async def test_response_quality(self):
        await AgentEvaluator.evaluate(
            agent_module="agent.agent",
            eval_dataset_file_path_or_dir="tests/eval/ritecare_responses.test.json",
            num_runs=2,
        )


class TestRBAC:
    """
    Validates that a field_officer can access BU5 tools.

    The agent should call BU5 tools like list_patient_visits and search_care_documents.
    Field officer session state is injected via initial_session_field_officer.json.
    """

    async def test_field_officer_allowed_tools(self):
        """Field officer asking about patient visits should succeed with BU5 tools."""
        await AgentEvaluator.evaluate(
            agent_module="agent.agent",
            eval_dataset_file_path_or_dir="tests/eval/ritecare_rbac.test.json",
            num_runs=2,
            initial_session_file="tests/eval/initial_session_field_officer.json",
        )
