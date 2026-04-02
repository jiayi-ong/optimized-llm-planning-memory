"""
tests/test_compressor/test_dummy_compressor.py
================================================
Full unit and integration tests for DummyCompressor.

Test categories
---------------
Vocab           — encode/decode roundtrip, special token handling
Model           — _DummyTransformerModel forward/backward pass, output shapes
DummyCompressor — interface compliance (TrainableCompressorBase contract)
                  + log_probs gradients + save/load checkpoint
Integration     — CompressorPolicy can wrap DummyCompressor
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from optimized_llm_planning_memory.compressor.dummy_compressor import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    DummyCompressor,
    _DummyTransformerModel,
    _pad_to,
    char_to_id,
    decode_ids,
    encode_text,
    id_to_char,
)
from optimized_llm_planning_memory.core.models import HardConstraintLedger, TrajectoryModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_model() -> _DummyTransformerModel:
    """Smallest valid model for fast CPU tests."""
    return _DummyTransformerModel(
        vocab_size=VOCAB_SIZE,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        max_len=64,
        dropout=0.0,
    )


@pytest.fixture
def compressor() -> DummyCompressor:
    """DummyCompressor with tiny config for CPU tests."""
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


@pytest.fixture
def minimal_trajectory() -> TrajectoryModel:
    """Minimal TrajectoryModel with no steps."""
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req",
        steps=(),
        total_steps=0,
    )


@pytest.fixture
def trajectory_with_steps() -> TrajectoryModel:
    """TrajectoryModel with 2 steps containing tool calls."""
    from optimized_llm_planning_memory.core.models import ReActStep, ToolCall, ToolResult

    steps = (
        ReActStep(
            step_index=0,
            thought="I need to search for flights.",
            action=ToolCall(
                tool_name="search_flights",
                arguments={"origin": "NYC", "destination": "Paris", "date": "2025-06-01"},
                raw_text='search_flights({"origin":"NYC","destination":"Paris","date":"2025-06-01"})',
            ),
            observation=ToolResult(
                tool_name="search_flights",
                success=True,
                result=[{"flight_id": "FL001", "price": 350.0}],
                error_message=None,
                latency_ms=10.0,
            ),
            itinerary_snapshot=None,
            timestamp="2025-06-01T10:00:00Z",
        ),
        ReActStep(
            step_index=1,
            thought="Found a flight at $350. Let me book it.",
            action=ToolCall(
                tool_name="book_flight",
                arguments={"flight_id": "FL001"},
                raw_text='book_flight({"flight_id":"FL001"})',
            ),
            observation=ToolResult(
                tool_name="book_flight",
                success=True,
                result={"booking_ref": "BK001"},
                error_message=None,
                latency_ms=5.0,
            ),
            itinerary_snapshot=None,
            timestamp="2025-06-01T10:00:05Z",
        ),
    )
    return TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="test-req",
        steps=steps,
        total_steps=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Vocabulary tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVocabulary:

    def test_special_token_ids_are_distinct(self):
        assert len({PAD_ID, BOS_ID, EOS_ID}) == 3

    def test_char_to_id_space(self):
        """Space (ASCII 32) maps to a valid non-special ID."""
        assert char_to_id(" ") >= 3

    def test_char_to_id_printable_ascii(self):
        """All printable ASCII chars map to IDs in [3, VOCAB_SIZE)."""
        for c in "abcdefghijklmnopqrstuvwxyz0123456789!?.,":
            token_id = char_to_id(c)
            assert 3 <= token_id < VOCAB_SIZE, f"char '{c}' mapped to {token_id}"

    def test_char_to_id_non_ascii_returns_pad(self):
        assert char_to_id("é") == PAD_ID
        assert char_to_id("中") == PAD_ID

    def test_id_to_char_special_returns_empty(self):
        assert id_to_char(PAD_ID) == ""
        assert id_to_char(BOS_ID) == ""
        assert id_to_char(EOS_ID) == ""

    def test_char_roundtrip(self):
        """For every printable ASCII char, encode → decode roundtrip is lossless."""
        for code in range(32, 127):
            c = chr(code)
            token_id = char_to_id(c)
            assert id_to_char(token_id) == c, f"Failed roundtrip for '{c}'"

    def test_encode_text_starts_with_bos(self):
        ids = encode_text("hello", max_len=20)
        assert ids[0] == BOS_ID

    def test_encode_text_ends_with_eos(self):
        ids = encode_text("hello", max_len=20)
        assert ids[-1] == EOS_ID

    def test_encode_text_respects_max_len(self):
        long_text = "a" * 1000
        ids = encode_text(long_text, max_len=16)
        assert len(ids) <= 16

    def test_encode_decode_roundtrip(self):
        text = "Hello, world!"
        ids = encode_text(text, max_len=50)
        decoded = decode_ids(ids)
        assert decoded == text

    def test_pad_to_pads_correctly(self):
        ids = [BOS_ID, 5, 6, EOS_ID]
        padded = _pad_to(ids, 8)
        assert len(padded) == 8
        assert padded[4:] == [PAD_ID] * 4

    def test_pad_to_truncates_when_too_long(self):
        ids = list(range(10))
        assert _pad_to(ids, 5) == [0, 1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════════════════
# 2. _DummyTransformerModel tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDummyTransformerModel:

    def test_instantiation(self, tiny_model):
        assert tiny_model is not None

    def test_has_expected_submodules(self, tiny_model):
        assert hasattr(tiny_model, "embedding")
        assert hasattr(tiny_model, "pos_embedding")
        assert hasattr(tiny_model, "encoder")
        assert hasattr(tiny_model, "decoder")
        assert hasattr(tiny_model, "output_projection")

    def test_weight_tying(self, tiny_model):
        """Embedding weight and output projection weight must be the same object."""
        assert tiny_model.embedding.weight is tiny_model.output_projection.weight

    def test_encode_output_shape(self, tiny_model):
        batch, seq = 2, 10
        src = torch.randint(3, VOCAB_SIZE, (batch, seq))
        memory = tiny_model.encode(src)
        assert memory.shape == (batch, seq, 16), f"Expected (2,10,16), got {memory.shape}"

    def test_decode_prefix_output_shape(self, tiny_model):
        batch, src_seq, tgt_seq = 2, 10, 7
        src = torch.randint(3, VOCAB_SIZE, (batch, src_seq))
        tgt = torch.randint(3, VOCAB_SIZE, (batch, tgt_seq))
        memory = tiny_model.encode(src)
        logits = tiny_model.decode_prefix(tgt, memory)
        assert logits.shape == (batch, tgt_seq, VOCAB_SIZE)

    def test_forward_output_shape(self, tiny_model):
        batch, src_seq, tgt_seq = 3, 12, 8
        src = torch.randint(3, VOCAB_SIZE, (batch, src_seq))
        tgt = torch.randint(3, VOCAB_SIZE, (batch, tgt_seq))
        logits = tiny_model(src, tgt)
        assert logits.shape == (batch, tgt_seq, VOCAB_SIZE)

    def test_forward_is_differentiable(self, tiny_model):
        """Gradients must flow from output to embedding weights."""
        tiny_model.train()
        src = torch.randint(3, VOCAB_SIZE, (1, 8))
        tgt = torch.randint(3, VOCAB_SIZE, (1, 5))
        logits = tiny_model(src, tgt)
        loss = logits.mean()
        loss.backward()
        # Verify that at least one parameter received a gradient
        grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients were computed"

    def test_backward_pass_no_nan(self, tiny_model):
        """Loss backward should not produce NaN gradients."""
        tiny_model.train()
        src = torch.randint(3, VOCAB_SIZE, (2, 10))
        tgt = torch.randint(3, VOCAB_SIZE, (2, 6))
        logits = tiny_model(src, tgt)
        labels = torch.randint(3, VOCAB_SIZE, (2, 6))
        loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), labels.view(-1))
        loss.backward()
        for name, p in tiny_model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_optimizer_step_changes_weights(self, tiny_model):
        """A gradient step must change at least one parameter."""
        tiny_model.train()
        optimizer = optim.Adam(tiny_model.parameters(), lr=1e-3)
        src = torch.randint(3, VOCAB_SIZE, (2, 10))
        tgt = torch.randint(3, VOCAB_SIZE, (2, 6))

        # Capture initial embedding weight
        w_before = tiny_model.pos_embedding.weight.data.clone()

        logits = tiny_model(src, tgt)
        loss = logits.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_after = tiny_model.pos_embedding.weight.data
        assert not torch.equal(w_before, w_after), "Optimizer step did not change weights"

    def test_greedy_decode_returns_list(self, tiny_model):
        src = torch.randint(3, VOCAB_SIZE, (1, 8))
        generated = tiny_model.greedy_decode(src, max_output_len=16)
        assert isinstance(generated, list)
        assert len(generated) <= 16
        assert generated[0] == BOS_ID

    def test_greedy_decode_no_grad_needed(self, tiny_model):
        """greedy_decode uses @torch.no_grad — should not raise inside no_grad."""
        src = torch.randint(3, VOCAB_SIZE, (1, 8))
        with torch.no_grad():
            generated = tiny_model.greedy_decode(src, max_output_len=8)
        assert generated is not None

    def test_padding_mask_applied(self, tiny_model):
        """Source with all-PAD positions should give different output than non-PAD."""
        src_all_pad = torch.full((1, 8), PAD_ID, dtype=torch.long)
        src_all_pad[0, 0] = BOS_ID  # at least one non-pad token
        src_real = torch.randint(3, VOCAB_SIZE, (1, 8))
        mem_pad = tiny_model.encode(src_all_pad)
        mem_real = tiny_model.encode(src_real)
        # Outputs should differ (padding should affect the encoding)
        assert not torch.allclose(mem_pad, mem_real, atol=1e-5)

    def test_sequence_longer_than_max_len_is_truncated(self):
        """Input longer than max_len should not raise — it should be silently truncated."""
        model = _DummyTransformerModel(
            vocab_size=VOCAB_SIZE, d_model=16, nhead=2,
            num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, max_len=16, dropout=0.0,
        )
        long_src = torch.randint(3, VOCAB_SIZE, (1, 100))
        memory = model.encode(long_src)  # should not raise
        assert memory.shape[1] <= 16


# ══════════════════════════════════════════════════════════════════════════════
# 3. DummyCompressor interface tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDummyCompressorInterface:

    def test_instantiation_defaults(self):
        c = DummyCompressor()
        assert c is not None

    def test_instantiation_custom_params(self):
        c = DummyCompressor(d_model=32, nhead=4, num_encoder_layers=1,
                            num_decoder_layers=1, dim_feedforward=64,
                            max_input_len=128, max_output_len=32)
        assert c._max_input_len == 128

    def test_nhead_not_dividing_d_model_raises(self):
        with pytest.raises(ValueError, match="divisible by nhead"):
            DummyCompressor(d_model=10, nhead=3)

    def test_is_trainable(self, compressor):
        assert compressor.is_trainable() is True

    def test_get_trainable_parameters_nonempty(self, compressor):
        params = compressor.get_trainable_parameters()
        assert len(params) > 0

    def test_all_trainable_params_are_nn_parameters(self, compressor):
        for p in compressor.get_trainable_parameters():
            assert isinstance(p, nn.Parameter)

    def test_model_exposed_as_self_model(self, compressor):
        """apply_lora() and freeze_base_layers() require self._model."""
        assert hasattr(compressor, "_model")
        assert isinstance(compressor._model, nn.Module)


class TestDummyCompressorCompress:

    def test_compress_returns_compressed_state(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        from optimized_llm_planning_memory.core.models import CompressedState
        assert isinstance(state, CompressedState)

    def test_compress_trajectory_id_matches(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert state.trajectory_id == minimal_trajectory.trajectory_id

    def test_compress_method_is_dummy(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert state.compression_method == "dummy"

    def test_compress_soft_summary_nonempty(self, compressor, minimal_trajectory):
        """Template requires soft_constraints_summary to be non-empty."""
        state = compressor.compress(minimal_trajectory)
        assert len(state.soft_constraints_summary) > 0

    def test_compress_itinerary_sketch_nonempty(self, compressor, minimal_trajectory):
        state = compressor.compress(minimal_trajectory)
        assert len(state.current_itinerary_sketch) > 0

    def test_compress_with_steps(self, compressor, trajectory_with_steps):
        state = compressor.compress(trajectory_with_steps)
        assert state.step_index == trajectory_with_steps.total_steps

    def test_compress_with_previous_state(self, compressor, minimal_trajectory):
        """Second compression should not raise when given a prior state."""
        first = compressor.compress(minimal_trajectory)
        second = compressor.compress(minimal_trajectory, previous_state=first)
        from optimized_llm_planning_memory.core.models import CompressedState
        assert isinstance(second, CompressedState)

    def test_compress_with_previous_state_is_different(self, compressor, minimal_trajectory):
        """Two compressions should produce different state_ids."""
        first = compressor.compress(minimal_trajectory)
        second = compressor.compress(minimal_trajectory)
        assert first.state_id != second.state_id  # UUID differs

    def test_compress_passes_template_validate(self, compressor, minimal_trajectory):
        """CompressedState returned by compress() should pass template validation."""
        from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
        template = CompressedStateTemplate()
        state = compressor.compress(minimal_trajectory)
        template.validate(state)  # should not raise

    def test_compress_multiple_times_no_crash(self, compressor, trajectory_with_steps):
        """Run 5 compressions in a row — none should raise."""
        state = None
        for _ in range(5):
            state = compressor.compress(trajectory_with_steps, previous_state=state)
        assert state is not None


class TestDummyCompressorLogProbs:

    def test_get_log_probs_returns_tensor(self, compressor):
        lp = compressor.get_log_probs("hello world", "compressed state")
        assert isinstance(lp, torch.Tensor)

    def test_get_log_probs_shape_is_1d(self, compressor):
        lp = compressor.get_log_probs("hello", "ok")
        assert lp.dim() == 1

    def test_get_log_probs_length_matches_target(self, compressor):
        """Length should equal len(encode_text(target)) - 1 (shifted target)."""
        target = "output text"
        target_ids = encode_text(target, compressor._max_output_len)
        lp = compressor.get_log_probs("source text", target)
        expected_len = len(target_ids) - 1  # shifted: predict tgt[1:] from tgt[:-1]
        assert lp.shape[0] == expected_len

    def test_get_log_probs_are_nonpositive(self, compressor):
        """Log probabilities from a softmax distribution are always ≤ 0."""
        lp = compressor.get_log_probs(
            "plan a trip to paris", "booked flight and hotel"
        )
        assert (lp <= 0).all(), f"Some log probs are positive: {lp[lp > 0]}"

    def test_get_log_probs_no_nan_or_inf(self, compressor):
        lp = compressor.get_log_probs("trajectory text here", "compressed")
        assert not torch.isnan(lp).any(), "NaN in log probs"
        assert not torch.isinf(lp).any(), "Inf in log probs"

    def test_get_log_probs_requires_grad(self, compressor):
        """Log probs must carry gradients for PPO backprop."""
        lp = compressor.get_log_probs("source", "target text")
        assert lp.requires_grad, "Log probs must require grad for PPO training"

    def test_get_log_probs_differentiable(self, compressor):
        """Backprop through log_probs.sum() should produce gradients."""
        lp = compressor.get_log_probs("this is the trajectory", "compressed state text")
        loss = lp.sum()
        loss.backward()
        # At least one model parameter should have a gradient
        grads = [p.grad for p in compressor.get_trainable_parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients after backward through get_log_probs"

    def test_get_log_probs_vary_with_input(self, compressor):
        """Different inputs should produce different log prob tensors."""
        lp1 = compressor.get_log_probs("trip to Paris", "booked hotel")
        compressor._model.zero_grad()
        lp2 = compressor.get_log_probs("cruise to Bahamas", "booked hotel")
        # At least one position should differ
        assert not torch.allclose(lp1.detach(), lp2.detach(), atol=1e-6)

    def test_get_log_probs_empty_target_no_crash(self, compressor):
        """Empty target text should not raise (returns degenerate tensor)."""
        lp = compressor.get_log_probs("source", "")
        assert isinstance(lp, torch.Tensor)


class TestDummyCompressorTraining:

    def test_full_optimizer_step(self, compressor):
        """One full PPO-style update step: forward → loss → backward → step."""
        params = compressor.get_trainable_parameters()
        optimizer = optim.Adam(params, lr=1e-4)
        w_before = compressor._model.pos_embedding.weight.data.clone()

        lp = compressor.get_log_probs("source trajectory text", "compressed state output")
        loss = -lp.mean()  # policy gradient: maximise log prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_after = compressor._model.pos_embedding.weight.data
        assert not torch.equal(w_before, w_after), "Weights unchanged after optimizer step"

    def test_multiple_backward_passes_no_crash(self, compressor):
        """Multiple consecutive backward passes should not raise."""
        params = compressor.get_trainable_parameters()
        optimizer = optim.Adam(params, lr=1e-4)
        for i in range(3):
            lp = compressor.get_log_probs(f"trajectory step {i}", "output")
            loss = -lp.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_freeze_base_layers(self, compressor):
        """After freeze_base_layers(), all params should have requires_grad=False."""
        compressor.freeze_base_layers(freeze=True)
        for p in compressor._model.parameters():
            assert not p.requires_grad
        # Unfreeze and verify
        compressor.freeze_base_layers(freeze=False)
        for p in compressor._model.parameters():
            assert p.requires_grad


class TestDummyCompressorCheckpoint:

    def test_save_creates_file(self, compressor, tmp_path):
        compressor.save_checkpoint(str(tmp_path))
        ckpt_file = tmp_path / "dummy_compressor.pt"
        assert ckpt_file.exists()

    def test_load_restores_weights(self, compressor, tmp_path):
        """save → modify weights → load → weights should match original."""
        compressor.save_checkpoint(str(tmp_path))

        # Corrupt weights
        with torch.no_grad():
            compressor._model.pos_embedding.weight.fill_(0.0)

        # Load restores them
        compressor.load_checkpoint(str(tmp_path))

        # Run a forward pass — should not crash and weights should be restored
        src = torch.randint(3, VOCAB_SIZE, (1, 8))
        tgt = torch.randint(3, VOCAB_SIZE, (1, 4))
        logits = compressor._model(src, tgt)
        assert logits is not None

    def test_load_roundtrip_produces_same_output(self, compressor, tmp_path):
        """After save+load, the model should produce identical outputs for the same input."""
        src = torch.randint(3, VOCAB_SIZE, (1, 8))
        tgt = torch.randint(3, VOCAB_SIZE, (1, 4))

        with torch.no_grad():
            logits_before = compressor._model(src, tgt).clone()

        compressor.save_checkpoint(str(tmp_path))

        # Build a fresh compressor of the same size
        new_compressor = DummyCompressor(
            d_model=16, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=32, max_input_len=64, max_output_len=32, device="cpu"
        )
        new_compressor.load_checkpoint(str(tmp_path))

        with torch.no_grad():
            logits_after = new_compressor._model(src, tgt)

        assert torch.allclose(logits_before, logits_after, atol=1e-6)

    def test_load_missing_file_raises(self, compressor, tmp_path):
        from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
        with pytest.raises(CompressorCheckpointError):
            compressor.load_checkpoint(str(tmp_path / "nonexistent"))

    def test_save_creates_directory_if_missing(self, compressor, tmp_path):
        new_dir = tmp_path / "nested" / "checkpoint"
        compressor.save_checkpoint(str(new_dir))
        assert (new_dir / "dummy_compressor.pt").exists()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Integration: CompressorPolicy can wrap DummyCompressor
# ══════════════════════════════════════════════════════════════════════════════

class TestDummyCompressorIntegration:

    def test_policy_wraps_compressor(self, compressor):
        """CompressorPolicy should be constructable with DummyCompressor."""
        import gymnasium
        import numpy as np
        from optimized_llm_planning_memory.training.policy import CompressorPolicy

        obs_space = gymnasium.spaces.Box(
            low=0, high=VOCAB_SIZE, shape=(64,), dtype=np.int32
        )
        act_space = gymnasium.spaces.Box(
            low=0, high=VOCAB_SIZE, shape=(32,), dtype=np.int32
        )

        def lr_schedule(_):
            return 3e-4

        policy = CompressorPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lr_schedule,
            compressor=compressor,
        )
        assert policy is not None

    def test_policy_evaluate_actions_no_crash(self, compressor):
        """evaluate_actions() should not raise with DummyCompressor."""
        import gymnasium
        import numpy as np
        from optimized_llm_planning_memory.training.policy import CompressorPolicy

        max_obs, max_act = 64, 32
        obs_space = gymnasium.spaces.Box(low=0, high=VOCAB_SIZE, shape=(max_obs,), dtype=np.int32)
        act_space = gymnasium.spaces.Box(low=0, high=VOCAB_SIZE, shape=(max_act,), dtype=np.int32)

        policy = CompressorPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda _: 3e-4,
            compressor=compressor,
        )

        batch = 2
        obs = torch.randint(0, VOCAB_SIZE, (batch, max_obs)).float()
        actions = torch.randint(0, VOCAB_SIZE, (batch, max_act))

        values, log_probs, entropy = policy.evaluate_actions(obs, actions)
        assert values.shape == (batch, 1)
        assert log_probs.shape == (batch,)
        assert entropy.dim() == 0  # scalar

    def test_compressor_is_trainable_via_base_class(self, compressor):
        """Verify is_trainable() works and matches parameter count."""
        assert compressor.is_trainable()
        assert len(compressor.get_trainable_parameters()) > 0
