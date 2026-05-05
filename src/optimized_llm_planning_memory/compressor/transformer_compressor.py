"""
compressor/transformer_compressor.py
=====================================
TransformerCompressor — trainable HuggingFace seq2seq/decoder compressor.

This is the PRIMARY compressor trained via PPO. It wraps any HuggingFace
encoder-decoder (T5, BART) or decoder-only (GPT-2, Phi, Qwen) model.

The model is prompted with the trajectory text and generates the compressed
state as a text sequence. The template parser reconstructs the structured
``CompressedState`` from the generated text.

Model selection guidance (from ``configs/compressor/transformer.yaml``)
------------------------------------------------------------------------
- ``google/flan-t5-small``  : ~80M params, fast, good instruction following.
  Recommended default for Colab T4 GPU.
- ``facebook/bart-base``    : ~140M params, strong summarisation baseline.
- ``Qwen/Qwen2.5-0.5B``    : ~500M modern decoder; use with LoRA on Colab A100.
- Any model on HuggingFace Hub works via ``model_name_or_path``.

PPO training notes
------------------
``get_log_probs()`` calls ``model.forward()`` with teacher-forcing to compute
token-level log-probabilities. This is what ``CompressorPolicy.evaluate_actions()``
calls to compute the PPO ratio r_t(θ).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel

from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.config import LoRAConfig, TransformerCompressorConfig as _Cfg
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


# Re-export the config alias used by TransformerCompressor
# (defined in core/config.py but also importable here for convenience)
try:
    from optimized_llm_planning_memory.core.config import CompressorConfig as TransformerCompressorConfig
except ImportError:
    TransformerCompressorConfig = None  # type: ignore[assignment, misc]


class TransformerCompressor(TrainableCompressorBase):
    """
    Trainable compressor backed by a HuggingFace transformer model.

    Parameters
    ----------
    model_name_or_path : HuggingFace model ID or local path.
    max_input_tokens   : Max tokenised length of the trajectory input.
    max_output_tokens  : Max tokenised length of the generated compression.
    device             : 'cpu' | 'cuda' | 'auto' (auto selects GPU if available).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        max_input_tokens: int = 2048,
        max_output_tokens: int = 512,
        device: str = "auto",
        use_lora: bool = False,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        self._model_name = model_name_or_path
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._device = self._resolve_device(device)
        self._template = CompressedStateTemplate()

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path
        ).to(self._device)

        # Auto-apply LoRA adapters when requested (L1 fix)
        if use_lora and lora_config is not None:
            self.apply_lora(lora_config)

    # ── CompressorBase ────────────────────────────────────────────────────────

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Generate a compressed state by running the model in inference mode.

        The input is the trajectory text (prefixed with a task prompt).
        The output is decoded and parsed via ``CompressedStateTemplate.parse()``.
        """
        input_text = self._build_input(trajectory, previous_state)
        input_ids = self._tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=self._max_input_tokens,
            truncation=True,
        ).to(self._device)

        raw_token_count = int(input_ids.shape[1])

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self._max_output_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        output_token_count = int(output_ids.shape[-1])
        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        try:
            state = self._template.parse(
                text=generated_text,
                trajectory_id=trajectory.trajectory_id,
                step_index=trajectory.total_steps,
                compression_method="transformer",
            )
        except Exception:
            state = self._make_fallback_state(
                trajectory, generated_text, previous_state
            )

        return state.model_copy(update={
            "token_count": output_token_count,
            "raw_token_count": raw_token_count,
        })

    # ── TrainableCompressorBase ───────────────────────────────────────────────

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Compute per-token log p(compressed_token | trajectory_text).

        Uses explicit teacher-forcing via ``decoder_input_ids`` rather than
        passing ``labels`` to the model. Passing ``labels`` triggers an internal
        cross-entropy loss computation that is redundant here and can obscure the
        exact log-prob semantics. With explicit decoder inputs:
          - decoder_input_ids[t] = target[t-1] (BOS at position 0)
          - logits[t] predicts target[t]
        Padding tokens are masked out of the returned log-probs.

        Returns
        -------
        torch.Tensor of shape (target_sequence_length,)
        """
        input_ids = self._tokenizer.encode(
            trajectory_text,
            return_tensors="pt",
            max_length=self._max_input_tokens,
            truncation=True,
        ).to(self._device)

        target_ids = self._tokenizer.encode(
            compressed_text,
            return_tensors="pt",
            max_length=self._max_output_tokens,
            truncation=True,
        ).to(self._device)  # (1, target_len)

        # Explicit teacher-forcing: decoder sees [BOS, target[0], ..., target[-2]]
        # so logits[t] predicts target[t].
        bos_id = self._model.config.decoder_start_token_id or 0
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=self._device)
        decoder_input_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)  # (1, target_len)

        outputs = self._model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits.squeeze(0)           # (target_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)    # (target_len, vocab_size)

        target = target_ids.squeeze(0)               # (target_len,)
        token_log_probs = log_probs.gather(
            dim=-1, index=target.unsqueeze(-1)
        ).squeeze(-1)                                # (target_len,)

        # Zero out padding positions so they don't distort the PPO log-prob sum
        pad_id = self._tokenizer.pad_token_id or 0
        non_pad_mask = (target != pad_id).float()
        return token_log_probs * non_pad_mask

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters with requires_grad=True."""
        return [p for p in self._model.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str) -> None:
        """Save model and tokenizer to ``path``."""
        try:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to save checkpoint to '{path}': {exc}"
            ) from exc

    def load_checkpoint(self, path: str) -> None:
        """Load model and tokenizer from ``path``."""
        import os
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            # PEFT-format checkpoint (saved after apply_lora): adapter_config.json present.
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                from peft import PeftModel
                base = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
                self._model = PeftModel.from_pretrained(base, path)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to load checkpoint from '{path}': {exc}"
            ) from exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_input(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None,
    ) -> str:
        parts = [
            "Compress the following travel planning trajectory into a structured memory state.",
            "",
        ]
        if previous_state is not None:
            from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
            parts.append("PREVIOUS STATE:")
            parts.append(CompressedStateTemplate().render(previous_state))
            parts.append("")
            parts.append("NEW STEPS:")
        parts.append(trajectory.to_text())
        return "\n".join(parts)

    def _make_fallback_state(
        self,
        trajectory: TrajectoryModel,
        raw_text: str,
        previous_state: CompressedState | None,
    ) -> CompressedState:
        """Create a minimal CompressedState when template parsing fails."""
        from optimized_llm_planning_memory.core.models import HardConstraintLedger
        prior_ledger = (
            previous_state.hard_constraint_ledger
            if previous_state else HardConstraintLedger(constraints=(), satisfied_ids=(), violated_ids=(), unknown_ids=())
        )
        return CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=prior_ledger,
            soft_constraints_summary="(parse failed — raw output below)",
            decisions_made=[],
            open_questions=[],
            key_discoveries=[],
            current_itinerary_sketch=raw_text[:500],
            compression_method="transformer",
            token_count=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
