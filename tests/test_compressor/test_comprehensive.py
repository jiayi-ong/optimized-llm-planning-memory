"""
tests/test_compressor/test_comprehensive.py
============================================
Comprehensive test suite for the compressor layer.

Coverage goals
--------------
- Abstract contract enforcement (CompressorBase, TrainableCompressorBase)
- IdentityCompressor: forward pass, gradient propagation, checkpoint I/O,
  metadata, and edge-cases not covered elsewhere
- Cross-compressor protocol compliance (parametrized fixture)
- CompressedStateTemplate edge-cases (multiline content, long text)
- Checkpoint error paths
- Slow-marked TransformerCompressor smoke test (run with -m slow)

Run with
--------
    pytest tests/test_compressor/test_comprehensive.py -v
    pytest tests/test_compressor/test_comprehensive.py -v -m slow  # includes GPU tests
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.compressor.dummy_compressor import DummyCompressor
from optimized_llm_planning_memory.compressor.identity_compressor import IdentityCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.exceptions import (
    CompressedStateRenderError,
    CompressorCheckpointError,
    LogProbsNotSupportedError,
)
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    ReActStep,
    ToolCall,
    ToolResult,
    TrajectoryModel,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_trajectory() -> TrajectoryModel:
    """Zero-step trajectory — the simplest valid input."""
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req-001",
        steps=(),
        total_steps=0,
    )


@pytest.fixture
def two_step_trajectory() -> TrajectoryModel:
    """Two-step trajectory with a successful tool call followed by a booking."""
    steps = (
        ReActStep(
            step_index=0,
            thought="Discover available cities.",
            action=ToolCall(
                tool_name="get_available_routes",
                arguments={},
                raw_text="get_available_routes({})",
            ),
            observation=ToolResult(
                tool_name="get_available_routes",
                success=True,
                result=[{"city_id": "city_aeloria_0000", "city_name": "Aeloria"}],
                error_message=None,
                latency_ms=42.0,
            ),
            itinerary_snapshot=None,
            timestamp="2026-06-01T09:00:00Z",
        ),
        ReActStep(
            step_index=1,
            thought="Search hotels in Aeloria for 4 nights.",
            action=ToolCall(
                tool_name="search_hotels",
                arguments={"city_id": "city_aeloria_0000", "min_stars": 3.0},
                raw_text='search_hotels({"city_id":"city_aeloria_0000","min_stars":3.0})',
            ),
            observation=ToolResult(
                tool_name="search_hotels",
                success=True,
                result=[{"hotel_id": "hotel_001", "name": "Grand Hotel", "total_cost": 500.0}],
                error_message=None,
                latency_ms=88.0,
            ),
            itinerary_snapshot=None,
            timestamp="2026-06-01T09:01:00Z",
        ),
    )
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req-001",
        steps=steps,
        total_steps=2,
    )


@pytest.fixture
def sample_state() -> CompressedState:
    """A fully-populated CompressedState for template tests."""
    ledger = HardConstraintLedger(
        constraints=[],
        satisfied_ids=[],
        violated_ids=[],
        unknown_ids=[],
    )
    return CompressedState(
        state_id=str(uuid.uuid4()),
        trajectory_id=str(uuid.uuid4()),
        step_index=2,
        hard_constraint_ledger=ledger,
        soft_constraints_summary="Prefers Mediterranean cuisine and cultural sites.",
        decisions_made=["Booked hotel: Aeloria Boutique Inn ($356)"],
        open_questions=["Need to book restaurant for day 2"],
        key_discoveries=["3 Mediterranean restaurants found; Oliva Terrace rated 4.6"],
        current_itinerary_sketch="Jun 1: Check-in Boutique Inn. Jun 2-5: TBD.",
        compression_method="identity",
        token_count=64,
        created_at="2026-06-01T10:00:00Z",
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _tiny_dummy() -> DummyCompressor:
    """Smallest DummyCompressor for fast CPU tests."""
    return DummyCompressor(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        max_input_len=64,
        max_output_len=32,
        device="cpu",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Abstract contract enforcement
# ══════════════════════════════════════════════════════════════════════════════


class TestAbstractContractEnforcement:
    """
    CompressorBase and TrainableCompressorBase must refuse direct instantiation
    and must enforce that subclasses implement all abstract methods.
    """

    def test_compressor_base_cannot_be_instantiated(self):
        """CompressorBase is abstract — direct instantiation must raise."""
        with pytest.raises(TypeError):
            CompressorBase()  # type: ignore[abstract]

    def test_trainable_base_cannot_be_instantiated(self):
        """TrainableCompressorBase is also abstract."""
        with pytest.raises(TypeError):
            TrainableCompressorBase()  # type: ignore[abstract]

    def test_subclass_without_compress_raises(self):
        """A subclass that doesn't implement compress() must fail at instantiation."""
        class IncompleteCompressor(CompressorBase):
            pass  # missing compress()

        with pytest.raises(TypeError):
            IncompleteCompressor()

    def test_trainable_subclass_without_log_probs_raises(self):
        """TrainableCompressorBase requires get_log_probs(); omitting it must raise."""
        class IncompleteTrainable(TrainableCompressorBase):
            def compress(self, traj, prev=None):
                pass  # missing get_log_probs, get_trainable_parameters, save/load_checkpoint

        with pytest.raises(TypeError):
            IncompleteTrainable()

    def test_minimal_valid_compressor_base_subclass(self):
        """A subclass implementing only compress() must instantiate and comply."""
        class MinimalCompressor(CompressorBase):
            def compress(self, traj, prev=None):
                return CompressedState(
                    state_id=str(uuid.uuid4()),
                    trajectory_id=traj.trajectory_id,
                    step_index=traj.total_steps,
                    hard_constraint_ledger=HardConstraintLedger(
                        constraints=[], satisfied_ids=[], violated_ids=[], unknown_ids=[]
                    ),
                    soft_constraints_summary="none",
                    decisions_made=[],
                    open_questions=[],
                    key_discoveries=[],
                    current_itinerary_sketch="",
                    compression_method="minimal",
                    token_count=0,
                    created_at="2026-01-01T00:00:00Z",
                )

        c = MinimalCompressor()
        assert not c.is_trainable()
        assert c.get_trainable_parameters() == []

    def test_base_get_log_probs_raises_log_probs_not_supported(self):
        """
        The default get_log_probs() on a non-trainable compressor must raise
        LogProbsNotSupportedError (not a generic Exception).
        """
        class MinimalCompressor(CompressorBase):
            def compress(self, traj, prev=None):
                return None  # type: ignore[return-value]

        c = MinimalCompressor()
        with pytest.raises(LogProbsNotSupportedError):
            c.get_log_probs("input", "output")


# ══════════════════════════════════════════════════════════════════════════════
# 2. IdentityCompressor
# ══════════════════════════════════════════════════════════════════════════════


class TestIdentityCompressor:
    """
    Full coverage for IdentityCompressor: the RAW-baseline compressor that
    passes the full trajectory through unchanged but still participates in
    the RL gradient loop via a single dummy parameter.
    """

    @pytest.fixture
    def compressor(self) -> IdentityCompressor:
        return IdentityCompressor()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def test_init_creates_dummy_param(self, compressor):
        assert hasattr(compressor, "_dummy_param")
        assert isinstance(compressor._dummy_param, nn.Parameter)

    def test_dummy_param_shape(self, compressor):
        assert compressor._dummy_param.shape == (1,)

    def test_dummy_param_initial_value_zero(self, compressor):
        assert compressor._dummy_param.item() == pytest.approx(0.0)

    def test_init_no_reward_predictor_by_default(self, compressor):
        assert compressor._reward_predictor is None

    def test_init_with_reward_predictor(self):
        """A custom object can be attached as a reward predictor."""
        mock_rp = object()
        c = IdentityCompressor(reward_predictor=mock_rp)
        assert c._reward_predictor is mock_rp

    # ── compress() ─────────────────────────────────────────────────────────────

    def test_compress_returns_compressed_state(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert isinstance(state, CompressedState)

    def test_compress_trajectory_id_matches(self, compressor, two_step_trajectory):
        state = compressor.compress(two_step_trajectory)
        assert state.trajectory_id == two_step_trajectory.trajectory_id

    def test_compress_step_index_equals_total_steps(self, compressor, two_step_trajectory):
        state = compressor.compress(two_step_trajectory)
        assert state.step_index == two_step_trajectory.total_steps

    def test_compress_method_is_identity(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert state.compression_method == "identity"

    def test_compress_full_text_in_sketch(self, compressor, two_step_trajectory):
        """The full trajectory text must appear in current_itinerary_sketch."""
        state = compressor.compress(two_step_trajectory)
        full_text = two_step_trajectory.to_text()
        assert full_text in state.current_itinerary_sketch

    def test_compress_decisions_made_has_one_entry_per_step(self, compressor, two_step_trajectory):
        """Each ReActStep should produce one entry in decisions_made."""
        state = compressor.compress(two_step_trajectory)
        assert len(state.decisions_made) == two_step_trajectory.total_steps

    def test_compress_ignores_previous_state(self, compressor, two_step_trajectory, sample_state):
        """Identity compressor is stateless; previous_state must be ignored."""
        state_a = compressor.compress(two_step_trajectory, previous_state=None)
        state_b = compressor.compress(two_step_trajectory, previous_state=sample_state)
        # Both must produce identical current_itinerary_sketch
        assert state_a.current_itinerary_sketch == state_b.current_itinerary_sketch

    def test_compress_empty_trajectory(self, compressor, minimal_trajectory):
        """Zero-step trajectory must not raise."""
        state = compressor.compress(minimal_trajectory)
        assert state.decisions_made == []

    # ── Trainability ──────────────────────────────────────────────────────────

    def test_is_trainable_true(self, compressor):
        assert compressor.is_trainable() is True

    def test_get_trainable_parameters_returns_list(self, compressor):
        params = compressor.get_trainable_parameters()
        assert isinstance(params, list)

    def test_get_trainable_parameters_has_one_element(self, compressor):
        assert len(compressor.get_trainable_parameters()) == 1

    def test_trainable_parameter_is_dummy_param(self, compressor):
        params = compressor.get_trainable_parameters()
        assert params[0] is compressor._dummy_param

    # ── get_log_probs() ───────────────────────────────────────────────────────

    def test_log_probs_returns_tensor(self, compressor):
        lp = compressor.get_log_probs("trajectory text", "compressed output text")
        assert isinstance(lp, torch.Tensor)

    def test_log_probs_shape_matches_token_count(self, compressor):
        compressed = "one two three four"
        lp = compressor.get_log_probs("input", compressed)
        assert lp.shape == (4,)

    def test_log_probs_single_token(self, compressor):
        lp = compressor.get_log_probs("input", "token")
        assert lp.shape == (1,)

    def test_log_probs_requires_grad(self, compressor):
        """Log-probs must be in the autograd graph for PPO to flow gradients."""
        lp = compressor.get_log_probs("input", "a b c")
        assert lp.requires_grad

    def test_log_probs_gradient_flows_to_dummy_param(self, compressor):
        """
        The PPO gradient loop calls loss.backward() and expects dummy_param.grad
        to be populated. This is the most critical correctness property.
        """
        lp = compressor.get_log_probs("trajectory", "a b c d e")
        loss = -lp.mean()  # mimic PPO policy gradient loss
        loss.backward()
        assert compressor._dummy_param.grad is not None
        assert compressor._dummy_param.grad.shape == (1,)

    def test_log_probs_values_are_negative(self, compressor):
        """Log-probabilities must be ≤ 0 (probabilities ≤ 1)."""
        lp = compressor.get_log_probs("input", "a b c d")
        assert (lp <= 0.0).all()

    def test_log_probs_empty_compressed_text_does_not_crash(self, compressor):
        """Empty compressed text should still return a tensor (length ≥ 1)."""
        lp = compressor.get_log_probs("input", "")
        assert lp.numel() >= 1

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def test_save_and_load_roundtrip(self, compressor):
        """Saving then loading must restore _dummy_param to its saved value."""
        compressor._dummy_param.data.fill_(3.14)
        with tempfile.TemporaryDirectory() as tmp:
            compressor.save_checkpoint(tmp)
            fresh = IdentityCompressor()
            fresh.load_checkpoint(tmp)
            assert fresh._dummy_param.item() == pytest.approx(3.14, abs=1e-5)

    def test_save_creates_checkpoint_file(self, compressor):
        with tempfile.TemporaryDirectory() as tmp:
            compressor.save_checkpoint(tmp)
            assert (Path(tmp) / "identity_compressor.pt").exists()

    def test_load_missing_checkpoint_raises(self, compressor):
        with pytest.raises(CompressorCheckpointError):
            compressor.load_checkpoint("/nonexistent/path/that/does/not/exist")

    def test_load_preserves_gradient_flow(self, compressor):
        """After loading, the gradient flow test must still pass."""
        with tempfile.TemporaryDirectory() as tmp:
            compressor.save_checkpoint(tmp)
            fresh = IdentityCompressor()
            fresh.load_checkpoint(tmp)
            lp = fresh.get_log_probs("t", "a b c")
            loss = -lp.mean()
            loss.backward()
            assert fresh._dummy_param.grad is not None

    # ── Metadata ──────────────────────────────────────────────────────────────

    def test_get_metadata_is_dict(self, compressor):
        assert isinstance(compressor.get_metadata(), dict)

    def test_get_metadata_type_is_identity(self, compressor):
        assert compressor.get_metadata()["type"] == "identity"

    def test_get_metadata_param_count(self, compressor):
        assert compressor.get_metadata()["param_count"] == 1

    def test_get_metadata_trainable_true(self, compressor):
        assert compressor.get_metadata()["trainable"] is True

    def test_get_metadata_reward_predictor_flag_false(self, compressor):
        assert compressor.get_metadata()["has_reward_predictor"] is False

    def test_get_metadata_reward_predictor_flag_true_when_attached(self):
        c = IdentityCompressor(reward_predictor=object())
        assert c.get_metadata()["has_reward_predictor"] is True

    def test_get_metadata_is_json_serialisable(self, compressor):
        import json
        # Must not raise
        json.dumps(compressor.get_metadata())


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cross-compressor protocol compliance
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(
    params=["identity", "dummy"],
    ids=["IdentityCompressor", "DummyCompressor"],
)
def any_trainable_compressor(request) -> TrainableCompressorBase:
    """
    Parametrized fixture that yields each trainable compressor in turn.
    Any test using this fixture runs against both implementations.
    """
    if request.param == "identity":
        return IdentityCompressor()
    return _tiny_dummy()


class TestTrainableProtocolCompliance:
    """
    Invariants that must hold for EVERY TrainableCompressorBase implementation.
    Adding a new compressor? Add it to the parametrized fixture above and
    all tests here run for free.
    """

    def test_is_trainable_returns_bool(self, any_trainable_compressor):
        result = any_trainable_compressor.is_trainable()
        assert isinstance(result, bool)

    def test_is_trainable_true(self, any_trainable_compressor):
        """Every trainable compressor must report is_trainable() == True."""
        assert any_trainable_compressor.is_trainable() is True

    def test_get_trainable_parameters_nonempty(self, any_trainable_compressor):
        params = any_trainable_compressor.get_trainable_parameters()
        assert len(params) > 0

    def test_get_trainable_parameters_are_nn_parameters(self, any_trainable_compressor):
        for p in any_trainable_compressor.get_trainable_parameters():
            assert isinstance(p, nn.Parameter)

    def test_compress_returns_compressed_state(self, any_trainable_compressor, minimal_trajectory):
        state = any_trainable_compressor.compress(minimal_trajectory)
        assert isinstance(state, CompressedState)

    def test_compress_trajectory_id_propagated(self, any_trainable_compressor, minimal_trajectory):
        state = any_trainable_compressor.compress(minimal_trajectory)
        assert state.trajectory_id == minimal_trajectory.trajectory_id

    def test_compress_returns_valid_compression_method_string(
        self, any_trainable_compressor, minimal_trajectory
    ):
        state = any_trainable_compressor.compress(minimal_trajectory)
        assert isinstance(state.compression_method, str)
        assert len(state.compression_method) > 0

    def test_get_log_probs_returns_tensor(self, any_trainable_compressor):
        lp = any_trainable_compressor.get_log_probs("source text", "output text")
        assert isinstance(lp, torch.Tensor)

    def test_get_log_probs_1d(self, any_trainable_compressor):
        lp = any_trainable_compressor.get_log_probs("source text", "a b c d")
        assert lp.dim() == 1

    def test_get_log_probs_nonempty(self, any_trainable_compressor):
        lp = any_trainable_compressor.get_log_probs("source text", "output")
        assert lp.numel() >= 1

    def test_get_log_probs_values_non_positive(self, any_trainable_compressor):
        lp = any_trainable_compressor.get_log_probs("source text", "a b c")
        assert (lp <= 0.0 + 1e-6).all()

    def test_backward_populates_grads(self, any_trainable_compressor):
        """The full gradient path from log_probs to parameters must be intact."""
        lp = any_trainable_compressor.get_log_probs("source text", "a b c d")
        loss = -lp.mean()
        loss.backward()
        params = any_trainable_compressor.get_trainable_parameters()
        grads_populated = any(p.grad is not None for p in params)
        assert grads_populated, (
            "No trainable parameter received a gradient — gradient flow is broken."
        )

    def test_get_metadata_has_required_keys(self, any_trainable_compressor):
        meta = any_trainable_compressor.get_metadata()
        for key in ("type", "param_count", "trainable"):
            assert key in meta, f"get_metadata() is missing required key '{key}'"

    def test_get_metadata_trainable_consistent_with_is_trainable(
        self, any_trainable_compressor
    ):
        meta = any_trainable_compressor.get_metadata()
        assert meta["trainable"] == any_trainable_compressor.is_trainable()

    def test_save_load_checkpoint_roundtrip(self, any_trainable_compressor):
        """Save then load must not raise and must be consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            any_trainable_compressor.save_checkpoint(tmp)
            # Load into a fresh instance of the same type
            fresh = type(any_trainable_compressor).__new__(type(any_trainable_compressor))
            fresh.__init__() if type(any_trainable_compressor) == IdentityCompressor else (
                fresh := _tiny_dummy()
            )
            # Just verify load does not raise
            any_trainable_compressor.load_checkpoint(tmp)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CompressedStateTemplate edge-cases
# ══════════════════════════════════════════════════════════════════════════════


class TestTemplateEdgeCases:
    """
    Edge-cases for CompressedStateTemplate beyond the basic roundtrip already
    tested in test_template.py.
    """

    @pytest.fixture
    def template(self):
        return CompressedStateTemplate()

    def test_render_produces_all_required_section_headers(self, template, sample_state):
        rendered = template.render(sample_state)
        for section in CompressedStateTemplate.REQUIRED_SECTIONS:
            assert f"## {section} ##" in rendered

    def test_render_multiline_decisions_preserved(self, template, sample_state):
        updated = sample_state.model_copy(
            update={"decisions_made": ["Line one", "Line two", "Line three"]}
        )
        rendered = template.render(updated)
        parsed = template.parse(rendered)
        assert parsed.decisions_made == ["Line one", "Line two", "Line three"]

    def test_render_parse_long_itinerary_sketch(self, template, sample_state):
        long_sketch = "Day N: " + ("activity; " * 200)
        updated = sample_state.model_copy(update={"current_itinerary_sketch": long_sketch})
        rendered = template.render(updated)
        parsed = template.parse(rendered)
        assert parsed.current_itinerary_sketch.strip() == long_sketch.strip()

    def test_validate_passes_valid_state(self, template, sample_state):
        """validate() must not raise on a fully-populated state."""
        template.validate(sample_state)  # should not raise

    def test_validate_raises_on_empty_soft_constraints(self, template, sample_state):
        broken = sample_state.model_copy(update={"soft_constraints_summary": ""})
        with pytest.raises(CompressedStateRenderError):
            template.validate(broken)

    def test_validate_raises_on_empty_itinerary_sketch(self, template, sample_state):
        broken = sample_state.model_copy(update={"current_itinerary_sketch": ""})
        with pytest.raises(CompressedStateRenderError):
            template.validate(broken)

    def test_render_then_validate_does_not_raise(self, template, sample_state):
        """render() must produce output that passes validate()."""
        rendered_state = template.parse(template.render(sample_state))
        template.validate(rendered_state)  # should not raise

    def test_empty_decisions_list_is_valid(self, template, sample_state):
        """An empty decisions_made list is valid (no bookings yet)."""
        state = sample_state.model_copy(update={"decisions_made": []})
        template.validate(state)  # must not raise

    def test_special_characters_in_sketch_survive_roundtrip(self, template, sample_state):
        sketch = "Café & Résumé: cost = $1,200 (€1,100) — confirmed ✓"
        updated = sample_state.model_copy(update={"current_itinerary_sketch": sketch})
        rendered = template.render(updated)
        parsed = template.parse(rendered)
        assert parsed.current_itinerary_sketch.strip() == sketch.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Checkpoint error paths
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckpointErrorPaths:
    """
    The CompressorCheckpointError contract: every trainable compressor must
    raise CompressorCheckpointError (not a generic FileNotFoundError or IOError)
    when asked to load from a missing or invalid path.
    """

    def test_identity_load_missing_dir_raises_checkpoint_error(self):
        c = IdentityCompressor()
        with pytest.raises(CompressorCheckpointError):
            c.load_checkpoint("/no/such/dir/12345")

    def test_identity_load_corrupted_file_raises_checkpoint_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "identity_compressor.pt"
            bad_path.write_bytes(b"not a valid pytorch file")
            c = IdentityCompressor()
            with pytest.raises(CompressorCheckpointError):
                c.load_checkpoint(tmp)

    def test_dummy_load_missing_dir_raises_checkpoint_error(self):
        c = _tiny_dummy()
        with pytest.raises(CompressorCheckpointError):
            c.load_checkpoint("/no/such/dir/99999")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DummyCompressor gradient tests (fast CPU subset)
# ══════════════════════════════════════════════════════════════════════════════


class TestDummyCompressorGradients:
    """
    Gradient-flow tests for DummyCompressor's seq2seq transformer.
    These complement the model-level tests in test_dummy_compressor.py.
    """

    @pytest.fixture
    def compressor(self):
        return _tiny_dummy()

    def test_get_log_probs_grad_through_decoder(self, compressor):
        """
        Gradient must flow all the way from the loss through the decoder layers
        to at least one encoder parameter — confirming end-to-end autograd.
        """
        lp = compressor.get_log_probs("the agent searched for hotels", "hotel found")
        loss = -lp.mean()
        loss.backward()
        # At least one parameter in the underlying model must have a non-None grad
        params_with_grad = [
            p for p in compressor._model.parameters() if p.grad is not None
        ]
        assert len(params_with_grad) > 0

    def test_optimizer_step_changes_params(self, compressor):
        """
        A single SGD step must change at least one parameter value —
        confirming the gradient is non-zero and the update is applied.
        """
        params = compressor.get_trainable_parameters()
        before = [p.data.clone() for p in params]

        opt = torch.optim.SGD(params, lr=1.0)
        lp = compressor.get_log_probs("source", "a b c")
        loss = -lp.mean()
        loss.backward()
        opt.step()

        changed = any(
            not torch.equal(a, b.data) for a, b in zip(before, params)
        )
        assert changed, "No parameter changed after an optimizer step — gradient may be zero."

    def test_multiple_backward_passes_accumulate_grads(self, compressor):
        """
        Without zero_grad(), gradients accumulate across backward calls.
        This tests that the graph is constructed fresh each forward pass.
        """
        params = compressor.get_trainable_parameters()

        lp1 = compressor.get_log_probs("input one", "output one")
        (-lp1.mean()).backward()
        grad_after_first = [p.grad.clone() for p in params if p.grad is not None]

        lp2 = compressor.get_log_probs("input two", "output two")
        (-lp2.mean()).backward()
        grad_after_second = [p.grad.clone() for p in params if p.grad is not None]

        # Accumulated grad must differ from the first-pass grad
        any_changed = any(
            not torch.equal(g1, g2)
            for g1, g2 in zip(grad_after_first, grad_after_second)
        )
        assert any_changed


# ══════════════════════════════════════════════════════════════════════════════
# 7. Slow / integration tests (requires HuggingFace model download)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestTransformerCompressorSmoke:
    """
    Smoke tests for TransformerCompressor.

    These tests download the flan-t5-small weights (~300 MB) and run on CPU,
    so they are marked @pytest.mark.slow and excluded from the default CI run.

    Run with:   pytest -m slow tests/test_compressor/test_comprehensive.py
    """

    @pytest.fixture(scope="class")
    def compressor(self):
        from optimized_llm_planning_memory.compressor.transformer_compressor import (
            TransformerCompressor,
        )
        return TransformerCompressor(device="cpu")

    def test_compress_returns_compressed_state(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert isinstance(state, CompressedState)

    def test_get_log_probs_returns_tensor(self, compressor):
        lp = compressor.get_log_probs("source text here", "output summary")
        assert isinstance(lp, torch.Tensor)

    def test_is_trainable(self, compressor):
        assert compressor.is_trainable() is True

    def test_gradient_flows(self, compressor):
        lp = compressor.get_log_probs("source text here", "a b c")
        loss = -lp.mean()
        loss.backward()
        params_with_grad = [
            p for p in compressor.get_trainable_parameters() if p.grad is not None
        ]
        assert len(params_with_grad) > 0

    def test_save_load_roundtrip(self, compressor):
        with tempfile.TemporaryDirectory() as tmp:
            compressor.save_checkpoint(tmp)
            compressor.load_checkpoint(tmp)  # must not raise
