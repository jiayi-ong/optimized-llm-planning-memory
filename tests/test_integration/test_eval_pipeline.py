"""
tests/test_integration/test_eval_pipeline.py
============================================
Integration tests for the full agent→evaluation pipeline.

No real LLM calls are made — litellm.completion is patched with scripted
multi-step responses that exercise:
  - Multiple tool calls (search + book)
  - Redundancy detection
  - Tool failure handling
  - Deterministic evaluation (all metric keys + value ranges)
  - Aggregation across multiple episodes

The ``@pytest.mark.integration`` tests at the bottom actually call a real LLM
(groq/llama3-8b-8192 free tier). They are skipped if GROQ_API_KEY is unset.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from optimized_llm_planning_memory.agent.context_builder import ContextBuilder
from optimized_llm_planning_memory.agent.modes import AgentMode
from optimized_llm_planning_memory.agent.prompts import get_system_prompt
from optimized_llm_planning_memory.agent.react_agent import ReActAgent
from optimized_llm_planning_memory.core.config import AgentConfig
from optimized_llm_planning_memory.evaluation.deterministic import DeterministicEvaluator
from optimized_llm_planning_memory.evaluation.evaluator import Evaluator
from optimized_llm_planning_memory.tools.events import EventBus
from optimized_llm_planning_memory.tools.registry import ToolRegistry
from optimized_llm_planning_memory.tools.tracker import ToolCallTracker
from tests.test_integration.mock_simulator import MockSimulator, make_test_requests


# ── Test helpers ──────────────────────────────────────────────────────────────

EXPECTED_DET_KEYS = {
    "hard_constraint_ratio",
    "soft_constraint_score",
    "tool_efficiency",
    "tool_failure_rate",
    "avg_tool_latency_ms",
    "steps_per_episode",
    "budget_adherence",
    "logical_consistency",
}


def _make_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _build_agent(simulator: MockSimulator, mode: AgentMode = AgentMode.RAW,
                 max_steps: int = 10) -> ReActAgent:
    tracker = ToolCallTracker()
    event_bus = EventBus()
    registry = ToolRegistry.from_config(simulator=simulator, tracker=tracker, event_bus=event_bus)
    config = AgentConfig(
        mode=mode.value,
        llm_model_id="groq/llama3-8b-8192",
        max_steps=max_steps,
        compress_every_n_steps=999,  # disable compression for RAW tests
        temperature=0.0,
    )
    context_builder = ContextBuilder(
        system_prompt=get_system_prompt("v1"),
        tool_registry=registry,
        llm_model_id="groq/llama3-8b-8192",
    )
    return ReActAgent(
        llm_model_id="groq/llama3-8b-8192",
        tool_registry=registry,
        compressor=None,
        context_builder=context_builder,
        config=config,
        mode=mode,
    )


# Scripted LLM response sequences for a Paris trip
_PARIS_SCRIPT = [
    # Search flights
    _make_response(
        'Thought: Find flights from New York to Paris.\n'
        'Action: search_flights({"origin": "New York", "destination": "Paris", '
        '"date": "2025-06-01", "num_passengers": 1})'
    ),
    # Book the cheap flight
    _make_response(
        'Thought: BudgetFly at $220 fits budget. Book it.\n'
        'Action: book_flight({"flight_id": "FL_NEW_PAR_002", '
        '"passenger_details": {"num_passengers": 1}})'
    ),
    # Search hotels
    _make_response(
        'Thought: Find a hotel in Paris.\n'
        'Action: search_hotels({"city": "Paris", "check_in": "2025-06-01", '
        '"check_out": "2025-06-03", "num_guests": 1})'
    ),
    # Book boutique hotel
    _make_response(
        'Thought: Boutique Inn at $120/night ($240 total). Book it.\n'
        'Action: book_hotel({"hotel_id": "HTL_PAR_BOUTIQUE", '
        '"guest_details": {"num_guests": 1, "check_in": "2025-06-01", '
        '"check_out": "2025-06-03"}})'
    ),
    # Search activities
    _make_response(
        'Thought: Look for museum activities in Paris.\n'
        'Action: search_activities({"city": "Paris", "date": "2025-06-01", '
        '"category": "culture"})'
    ),
    # Done
    _make_response(
        "Thought: Total: $220 (flight) + $240 (hotel) + $18 (museum) = $478. "
        "Well under $1500 budget. Planning complete.\nAction: DONE"
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tool tracking tests
# ══════════════════════════════════════════════════════════════════════════════

class TestToolTracking:

    def test_tool_stats_present_in_episode_log(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]  # Paris
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        assert len(log.tool_stats) > 0

    def test_each_tool_call_recorded(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        tool_names = {s.tool_name for s in log.tool_stats}
        assert "search_flights" in tool_names
        assert "book_flight" in tool_names
        assert "search_hotels" in tool_names
        assert "book_hotel" in tool_names

    def test_success_counts_correct(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        stats = {s.tool_name: s for s in log.tool_stats}
        assert stats["search_flights"].success_count == 1
        assert stats["book_flight"].success_count == 1
        assert stats["book_hotel"].success_count == 1

    def test_no_redundant_calls_in_clean_episode(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        total_redundant = sum(s.redundant_call_count for s in log.tool_stats)
        assert total_redundant == 0

    def test_redundant_call_detected(self):
        """Calling search_hotels twice with the same args → 1 redundant call."""
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        search_hotels_call = _make_response(
            'Thought: Search hotels.\n'
            'Action: search_hotels({"city": "Paris", "check_in": "2025-06-01", '
            '"check_out": "2025-06-03", "num_guests": 1})'
        )
        script = iter([search_hotels_call, search_hotels_call,
                       _make_response("Thought: Done.\nAction: DONE")])
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        stats = {s.tool_name: s for s in log.tool_stats}
        assert stats["search_hotels"].redundant_call_count == 1

    def test_latency_recorded(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        for stat in log.tool_stats:
            assert stat.avg_latency_ms >= 0.0
            assert stat.total_latency_ms >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. Deterministic evaluation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterministicEvaluation:

    @pytest.fixture
    def paris_episode_log(self):
        sim = MockSimulator()
        agent = _build_agent(sim)
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            return agent.run_episode(request, sim), request

    def test_all_det_metric_keys_present(self, paris_episode_log):
        log, request = paris_episode_log
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)
        assert EXPECTED_DET_KEYS.issubset(scores.keys()), (
            f"Missing keys: {EXPECTED_DET_KEYS - scores.keys()}"
        )

    def test_det_metric_values_in_range(self, paris_episode_log):
        """All bounded metrics must be in [0, 1]. Unbounded ones must be ≥ 0."""
        log, request = paris_episode_log
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)
        bounded = EXPECTED_DET_KEYS - {"avg_tool_latency_ms", "steps_per_episode",
                                        "tool_failure_rate"}
        for key in bounded:
            assert 0.0 <= scores[key] <= 1.0, f"{key} = {scores[key]} out of [0, 1]"
        assert scores["avg_tool_latency_ms"] >= 0.0
        assert scores["steps_per_episode"] >= 0.0
        assert 0.0 <= scores["tool_failure_rate"] <= 1.0

    def test_steps_per_episode_matches_log(self, paris_episode_log):
        log, request = paris_episode_log
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)
        assert scores["steps_per_episode"] == float(log.total_steps)

    def test_tool_failure_rate_zero_for_clean_episode(self, paris_episode_log):
        """All mock tools succeed → failure_rate should be 0."""
        log, request = paris_episode_log
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)
        assert scores["tool_failure_rate"] == pytest.approx(0.0)

    def test_tool_efficiency_one_for_no_redundancy(self, paris_episode_log):
        """No redundant calls → efficiency = 1.0."""
        log, request = paris_episode_log
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)
        assert scores["tool_efficiency"] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Full Evaluator pipeline (deterministic_only=True)
# ══════════════════════════════════════════════════════════════════════════════

class TestFullEvaluatorPipeline:

    def _run_episode(self, request, responses):
        sim = MockSimulator()
        agent = _build_agent(sim)
        script = iter(responses)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            return agent.run_episode(request, sim)

    def test_evaluate_episode_returns_eval_result(self):
        from optimized_llm_planning_memory.core.models import EvalResult
        from optimized_llm_planning_memory.core.config import EvalConfig

        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        log = self._run_episode(request, script)

        evaluator = Evaluator(config=EvalConfig(deterministic_only=True))
        result = evaluator.evaluate_episode(log, request)
        assert isinstance(result, EvalResult)

    def test_eval_result_has_correct_request_id(self):
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        log = self._run_episode(request, script)

        evaluator = Evaluator()
        result = evaluator.evaluate_episode(log, request)
        assert result.request_id == request.request_id

    def test_eval_result_has_correct_agent_mode(self):
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        log = self._run_episode(request, script)

        evaluator = Evaluator()
        result = evaluator.evaluate_episode(log, request)
        assert result.agent_mode == AgentMode.RAW.value

    def test_overall_score_in_unit_interval(self):
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT)
        log = self._run_episode(request, script)

        evaluator = Evaluator()
        result = evaluator.evaluate_episode(log, request)
        assert 0.0 <= result.overall_score <= 1.0

    def test_evaluate_dataset_three_requests(self):
        """Run three different user requests and evaluate all deterministically."""
        requests = make_test_requests()
        all_scripts = [
            list(_PARIS_SCRIPT),
            [
                _make_response(
                    'Thought: Search for flights to Rome.\n'
                    'Action: search_flights({"origin": "London", "destination": "Rome", '
                    '"date": "2025-06-10", "num_passengers": 2})'
                ),
                _make_response("Thought: Done.\nAction: DONE"),
            ],
            [
                _make_response(
                    'Thought: Search for flights to Barcelona.\n'
                    'Action: search_flights({"origin": "Amsterdam", "destination": "Barcelona", '
                    '"date": "2025-06-20", "num_passengers": 3})'
                ),
                _make_response("Thought: Done.\nAction: DONE"),
            ],
        ]

        from optimized_llm_planning_memory.core.config import EvalConfig
        evaluator = Evaluator(config=EvalConfig(deterministic_only=True))
        logs = []
        for req, script in zip(requests, all_scripts):
            script_iter = iter(script)
            log = self._run_episode(req, script_iter)
            logs.append(log)

        results = evaluator.evaluate_dataset(logs, requests)
        assert len(results) == 3
        for result in results:
            assert 0.0 <= result.overall_score <= 1.0

    def test_aggregate_returns_mean_and_std(self):
        """aggregate() must return _mean and _std keys for every metric."""
        requests = make_test_requests()
        all_scripts = [
            list(_PARIS_SCRIPT),
            [
                _make_response('Thought: search.\nAction: get_city_info({"city": "Rome"})'),
                _make_response("Thought: Done.\nAction: DONE"),
            ],
            [
                _make_response('Thought: search.\nAction: get_city_info({"city": "Barcelona"})'),
                _make_response("Thought: Done.\nAction: DONE"),
            ],
        ]

        from optimized_llm_planning_memory.core.config import EvalConfig
        evaluator = Evaluator(config=EvalConfig(deterministic_only=True))
        logs = []
        for req, script in zip(requests, all_scripts):
            log = self._run_episode(req, iter(script))
            logs.append(log)

        results = evaluator.evaluate_dataset(logs, requests)
        agg = evaluator.aggregate(results)

        assert "overall_score_mean" in agg
        assert "overall_score_std" in agg
        for key in EXPECTED_DET_KEYS:
            assert f"{key}_mean" in agg, f"Missing {key}_mean in aggregate"
            assert f"{key}_std" in agg, f"Missing {key}_std in aggregate"

    def test_evaluate_dataset_length_mismatch_raises(self):
        requests = make_test_requests()[:2]
        evaluator = Evaluator()
        with pytest.raises(ValueError, match="same length"):
            evaluator.evaluate_dataset([], requests)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAW mode with DummyCompressor (should not compress)
# ══════════════════════════════════════════════════════════════════════════════

class TestRawModeWithDummyCompressor:

    def test_raw_mode_ignores_dummy_compressor(self):
        """Even if a compressor is provided, RAW mode must not produce compressed states."""
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor

        sim = MockSimulator()
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        compressor = DummyCompressor(d_model=16, nhead=2, num_encoder_layers=1,
                                      num_decoder_layers=1, dim_feedforward=32)
        config = AgentConfig(
            mode=AgentMode.RAW.value,
            llm_model_id="groq/llama3-8b-8192",
            max_steps=5,
            compress_every_n_steps=1,  # very frequent trigger if it fired
        )
        agent = ReActAgent(
            llm_model_id="groq/llama3-8b-8192",
            tool_registry=registry,
            compressor=compressor,
            context_builder=ContextBuilder(
                system_prompt=get_system_prompt("v1"),
                tool_registry=registry,
            ),
            config=config,
            mode=AgentMode.RAW,
        )
        request = make_test_requests()[0]
        script = iter(_PARIS_SCRIPT[:2])
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        assert len(log.compressed_states) == 0, "RAW mode must not produce compressed states"


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPRESSOR mode with DummyCompressor
# ══════════════════════════════════════════════════════════════════════════════

class TestCompressorModeWithDummy:

    def test_compressor_mode_produces_compressed_states(self):
        """After compress_every_n_steps, at least one CompressedState should exist."""
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor

        sim = MockSimulator()
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        compressor = DummyCompressor(d_model=16, nhead=2, num_encoder_layers=1,
                                      num_decoder_layers=1, dim_feedforward=32)
        config = AgentConfig(
            mode=AgentMode.COMPRESSOR.value,
            llm_model_id="groq/llama3-8b-8192",
            max_steps=10,
            compress_every_n_steps=2,  # compress every 2 steps
        )
        agent = ReActAgent(
            llm_model_id="groq/llama3-8b-8192",
            tool_registry=registry,
            compressor=compressor,
            context_builder=ContextBuilder(
                system_prompt=get_system_prompt("v1"),
                tool_registry=registry,
            ),
            config=config,
            mode=AgentMode.COMPRESSOR,
        )
        request = make_test_requests()[0]

        # 6 steps before DONE → at least 1 compression at step 2
        responses = [
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Paris"})'),
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Paris"})'),
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Rome"})'),
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Rome"})'),
            _make_response("Thought: Done.\nAction: DONE"),
        ]
        script = iter(responses)
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        assert len(log.compressed_states) >= 1, (
            "COMPRESSOR mode with dummy compressor should produce at least 1 compressed state"
        )

    def test_compressed_state_is_valid_structure(self):
        """Each CompressedState in the log must pass template validation."""
        from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
        from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate

        sim = MockSimulator()
        tracker = ToolCallTracker()
        event_bus = EventBus()
        registry = ToolRegistry.from_config(simulator=sim, tracker=tracker, event_bus=event_bus)
        compressor = DummyCompressor(d_model=16, nhead=2, num_encoder_layers=1,
                                      num_decoder_layers=1, dim_feedforward=32)
        config = AgentConfig(
            mode=AgentMode.COMPRESSOR.value,
            llm_model_id="groq/llama3-8b-8192",
            max_steps=6,
            compress_every_n_steps=2,
        )
        agent = ReActAgent(
            llm_model_id="groq/llama3-8b-8192",
            tool_registry=registry,
            compressor=compressor,
            context_builder=ContextBuilder(
                system_prompt=get_system_prompt("v1"),
                tool_registry=registry,
            ),
            config=config,
            mode=AgentMode.COMPRESSOR,
        )
        request = make_test_requests()[0]
        responses = [
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Paris"})'),
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Paris"})'),
            _make_response('Thought: Step.\nAction: get_city_info({"city": "Rome"})'),
            _make_response("Thought: Done.\nAction: DONE"),
        ]
        script = iter(responses)
        template = CompressedStateTemplate()
        with patch("litellm.completion", side_effect=lambda **kw: next(script)):
            log = agent.run_episode(request, sim)
        for cs in log.compressed_states:
            template.validate(cs)  # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# 6. Live LLM integration tests (require GROQ_API_KEY, skipped in CI)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live LLM integration test",
)
class TestLiveLLMIntegration:
    """
    These tests call the real Groq API (llama3-8b-8192, free tier).
    Run manually: pytest tests/test_integration/test_eval_pipeline.py -m integration -v

    Requires: GROQ_API_KEY environment variable (free at console.groq.com).
    """

    def test_live_single_episode_raw_mode(self):
        """Run one real episode in RAW mode and verify EpisodeLog structure."""
        sim = MockSimulator()
        agent = _build_agent(sim, mode=AgentMode.RAW, max_steps=8)
        request = make_test_requests()[0]

        log = agent.run_episode(request, sim)
        assert log is not None
        assert log.total_steps > 0
        assert log.agent_mode == AgentMode.RAW.value
        # The agent may or may not call tools, but the log must be structurally valid
        assert log.trajectory is not None

    def test_live_deterministic_eval_after_real_episode(self):
        """Run a real episode then evaluate it — all metric keys must be present."""
        sim = MockSimulator()
        agent = _build_agent(sim, mode=AgentMode.RAW, max_steps=6)
        request = make_test_requests()[0]

        log = agent.run_episode(request, sim)
        evaluator = DeterministicEvaluator()
        scores = evaluator.score(log, request)

        assert EXPECTED_DET_KEYS.issubset(scores.keys())
        for key in EXPECTED_DET_KEYS - {"avg_tool_latency_ms", "steps_per_episode",
                                          "tool_failure_rate"}:
            assert 0.0 <= scores[key] <= 1.0
