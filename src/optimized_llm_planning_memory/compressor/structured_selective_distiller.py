"""
compressor/structured_selective_distiller.py
=============================================
StructuredSelectiveDistiller (SSD) — Design 1: standalone, no-MCTS compressor.

Design: Section-aware distillation via cross-attention routing
--------------------------------------------------------------
The key limitation of ``TransformerCompressor`` (the plain T5 seq2seq baseline)
is that it treats the full trajectory as an undifferentiated token sequence.
A booking confirmation step and an error recovery step receive equal treatment.
The T5 decoder must implicitly learn to route content to the right template
section from sparse RL reward signals alone.

SSD makes the routing EXPLICIT:

    1. Encode each trajectory step independently (T5-small encoder + LoRA).
    2. Learn 5 query vectors, one per free-form template section.
    3. Each section query cross-attends to ALL step embeddings.
       → The attention weights are the routing policy: which steps matter for
         this section?
    4. The 5 section context vectors (+ constraint ledger embedding) are packed
       into a [6 × H] sequence and passed directly to the T5 decoder as its
       encoder output. The decoder generates the full 6-section template.

This separation of selection (cross-attention) from generation (T5 decoder)
makes the distillation interpretable and easier to train:
  - The cross-attention weights can be visualised for ablation analysis.
  - The decoder doesn't need to "remember" which step index contained which fact.
  - The RL reward gradient flows through both the decoder log-probs AND the
    attention weights, teaching the routing policy directly.

RL training
-----------
``get_log_probs(trajectory_text, compressed_text)`` runs the full forward pass
in training mode (gradients enabled) and returns per-token log-probs via
teacher-forcing. This plugs directly into ``CompressorPolicy.evaluate_actions()``
for the PPO clipping ratio computation.

Memory (Colab T4)
-----------------
  T5-small base (fp16, frozen): ~160 MB
  LoRA adapters (fp32, trainable): ~14 MB
  Section queries + temporal embeddings: ~0.5 MB
  Adam optimizer state: ~28 MB
  Training activations (batch=32): ~400 MB
  ──────────────────────────────────────
  Total: ~600 MB — well within T4's 16 GB.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.config import LoRAConfig
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)

if TYPE_CHECKING:
    pass

# Step-header pattern produced by TrajectoryModel.to_text()
_STEP_HEADER_RE = re.compile(r"(?=\[Step \d+\])")

# Section-specific generation prefix tokens injected at decoder start.
# This nudges the decoder to respect the template structure from token 1.
_TEMPLATE_GENERATION_PREFIX = (
    "## HARD_CONSTRAINT_LEDGER ##\n"
)

# Max steps retained from a trajectory. Sliding window: keep first N_ANCHOR
# "anchor" steps (establish context) + most recent N_RECENT steps (current state).
_MAX_STEPS = 48
_N_ANCHOR = 4
_N_RECENT = 44


class SectionQueryAttention(nn.Module):
    """
    5 learnable query vectors that cross-attend to step embeddings.

    Each query corresponds to one of the 5 free-form template sections:
      0 → SOFT_CONSTRAINTS_SUMMARY
      1 → DECISIONS_MADE
      2 → OPEN_QUESTIONS
      3 → KEY_DISCOVERIES
      4 → CURRENT_ITINERARY_SKETCH

    Output: 5 context vectors [5, hidden_dim], one per section.

    Design rationale
    ----------------
    The query vectors start random and are trained end-to-end via RL. Over
    training they specialise: query 1 (DECISIONS_MADE) will learn to attend
    strongly to steps containing booking confirmations; query 2 (OPEN_QUESTIONS)
    will attend to steps with partial information or unresolved tool outputs.

    These attention weights can be logged per-step during evaluation as a
    human-interpretable routing map.
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 4) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._head_dim = hidden_dim // num_heads

        # 5 learnable section queries (one per free-form section)
        self.queries = nn.Parameter(torch.randn(5, hidden_dim) * 0.02)

        # Standard MHA projections
        self._q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        step_embeddings: torch.Tensor,   # [N_steps, hidden_dim]
    ) -> torch.Tensor:                   # [5, hidden_dim]
        """
        Returns one context vector per section by attending over step embeddings.

        Attention weights are NOT returned here but can be extracted for analysis
        by calling ``forward_with_weights()`` instead.
        """
        context, _ = self.forward_with_weights(step_embeddings)
        return context

    def forward_with_weights(
        self,
        step_embeddings: torch.Tensor,   # [N_steps, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (context_vectors [5, H], attention_weights [5, N_steps]).
        Useful for attention visualisation and ablation analysis.
        """
        N = step_embeddings.size(0)
        H = self._hidden_dim
        n_heads = self._num_heads
        head_dim = self._head_dim
        scale = head_dim ** -0.5

        # Project queries, keys, values
        Q = self._q_proj(self.queries)          # [5, H]
        K = self._k_proj(step_embeddings)       # [N, H]
        V = self._v_proj(step_embeddings)       # [N, H]

        # Reshape for multi-head: [n_heads, seq, head_dim]
        Q = Q.view(5, n_heads, head_dim).permute(1, 0, 2)   # [H, 5, head_dim]
        K = K.view(N, n_heads, head_dim).permute(1, 0, 2)   # [H, N, head_dim]
        V = V.view(N, n_heads, head_dim).permute(1, 0, 2)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * scale   # [H, 5, N]
        attn_weights = F.softmax(attn_scores, dim=-1)             # [H, 5, N]

        context = torch.bmm(attn_weights, V)                      # [H, 5, head_dim]
        context = context.permute(1, 0, 2).reshape(5, H)         # [5, H]
        context = self._out_proj(context)
        context = self._norm(context + self.queries)              # residual

        # Average attention weights across heads for interpretability
        mean_weights = attn_weights.mean(dim=0)                   # [5, N]
        return context, mean_weights


class StructuredSelectiveDistiller(TrainableCompressorBase):
    """
    Section-aware trainable compressor for standalone (no-MCTS) ReAct trajectories.

    Core innovation: section query cross-attention routes step-level information
    to the semantically correct template sections before generation.

    Parameters
    ----------
    model_name_or_path : HuggingFace T5 model ID (default: google/flan-t5-small).
    max_step_tokens    : Max tokens per individual step (longer steps are truncated).
    max_output_tokens  : Max tokens in the generated compressed state.
    device             : 'cpu' | 'cuda' | 'auto'.
    use_lora           : Apply LoRA adapters to encoder + decoder attention layers.
    lora_config        : LoRA hyperparameters (r, alpha, dropout, target_modules).

    Architecture note
    -----------------
    The T5 encoder is used per-step (encoding each ReActStep independently).
    The T5 decoder's encoder_outputs cross-attention slot receives the packed
    [section_contexts (5×H) | ledger_embed (1×H)] sequence instead of the
    standard token-level encoder output. This is the architectural crux that
    separates section-level context routing from token-level generation.
    """

    # Compression method string stored in CompressedState
    _METHOD = "structured_selective"

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        max_step_tokens: int = 128,
        max_output_tokens: int = 512,
        device: str = "auto",
        use_lora: bool = True,
        lora_config: LoRAConfig | None = None,
    ) -> None:
        self._model_name = model_name_or_path
        self._max_step_tokens = max_step_tokens
        self._max_output_tokens = max_output_tokens
        self._device = self._resolve_device(device)
        self._template = CompressedStateTemplate()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path
        ).to(self._device)

        hidden_dim = self._model.config.d_model  # 512 for T5-small

        # Trainable section query attention module
        self._section_attention = SectionQueryAttention(
            hidden_dim=hidden_dim, num_heads=4
        ).to(self._device)

        # Learned temporal position embeddings for step index awareness.
        # Step 0 = oldest (first) step; step _MAX_STEPS-1 = most recent.
        self._temporal_embedding = nn.Embedding(
            _MAX_STEPS, hidden_dim
        ).to(self._device)
        nn.init.normal_(self._temporal_embedding.weight, std=0.02)

        # Linear projection to pack [5 section contexts + 1 ledger] → [6, H]
        # for the T5 decoder cross-attention slot.
        self._encoder_out_proj = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        ).to(self._device)

        # LoRA injection: fine-tunes encoder + decoder attention Q,V projections
        if use_lora:
            _cfg = lora_config or LoRAConfig()
            self.apply_lora(_cfg)

    # ── CompressorBase ────────────────────────────────────────────────────────

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Generate a structured CompressedState from a ReAct trajectory.

        Pipeline:
          1. Extract + window steps → text list
          2. Encode each step via T5 encoder → step embeddings [N, H]
          3. Section query cross-attention → 5 section contexts [5, H]
          4. Encode constraint ledger text → ledger embedding [1, H]
          5. Pack as fake encoder output [6, H]
          6. T5 decoder generates template text (greedy beam search)
          7. Parse + inject deterministic constraint ledger → CompressedState
        """
        step_texts = self._extract_step_texts(trajectory)
        step_embs = self._encode_steps(step_texts)                # [N, H]
        section_contexts = self._section_attention(step_embs)     # [5, H]
        ledger_text = self._build_ledger_text(previous_state)
        ledger_emb = self._encode_single_text(ledger_text)        # [1, H]

        encoder_out = self._pack_encoder_output(section_contexts, ledger_emb)  # [1, 6, H]

        prefix_ids = self._get_generation_prefix_ids()
        with torch.no_grad():
            output_ids = self._model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                decoder_input_ids=prefix_ids,
                max_new_tokens=self._max_output_tokens,
                min_new_tokens=4,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._parse_or_fallback(
            generated_text, trajectory, previous_state
        )

    # ── TrainableCompressorBase ───────────────────────────────────────────────

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Compute per-token log p(compressed_token | trajectory, section_context).

        Runs the full forward pass (encode → section attention → decoder) in
        training mode with teacher-forcing to match the generate() path exactly.

        Parameters
        ----------
        trajectory_text : Full trajectory text from ``TrajectoryModel.to_text()``.
        compressed_text : Rendered template text from ``CompressedStateTemplate.render()``.

        Returns
        -------
        torch.Tensor of shape (target_len,) — non-padding token log-probs.
        """
        # Split trajectory text back into individual step strings
        step_texts = _split_trajectory_text(trajectory_text)
        if not step_texts:
            # Fallback: treat the whole text as a single step
            step_texts = [trajectory_text]

        # Apply the same windowing as compress()
        step_texts = _apply_step_window(step_texts)

        step_embs = self._encode_steps(step_texts)                # [N, H]
        section_contexts = self._section_attention(step_embs)     # [5, H]

        # Ledger text: use the constraint ledger prefix in compressed_text if available
        ledger_text = _extract_ledger_text_from_template(compressed_text)
        ledger_emb = self._encode_single_text(ledger_text)        # [1, H]

        encoder_out = self._pack_encoder_output(section_contexts, ledger_emb)  # [1, 6, H]

        # Tokenise target
        target_ids = self._tokenizer.encode(
            compressed_text,
            return_tensors="pt",
            max_length=self._max_output_tokens,
            truncation=True,
        ).to(self._device)  # [1, target_len]

        # Teacher-forcing: decoder sees [BOS, t0, t1, ..., t_{L-2}]
        bos_id = self._model.config.decoder_start_token_id or 0
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=self._device)
        decoder_input_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)  # [1, target_len]

        outputs = self._model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits.squeeze(0)           # [target_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)    # [target_len, vocab]

        target = target_ids.squeeze(0)               # [target_len]
        token_log_probs = log_probs.gather(
            dim=-1, index=target.unsqueeze(-1)
        ).squeeze(-1)                                # [target_len]

        pad_id = self._tokenizer.pad_token_id or 0
        non_pad_mask = (target != pad_id).float()
        return token_log_probs * non_pad_mask

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Return all trainable parameters:
          - LoRA adapter parameters (if LoRA is applied, base weights are frozen)
          - Section query attention parameters (always trainable)
          - Temporal position embeddings (always trainable)
          - Encoder output projection (always trainable)
        """
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        trainable += list(self._section_attention.parameters())
        trainable += list(self._temporal_embedding.parameters())
        trainable += list(self._encoder_out_proj.parameters())
        return trainable

    def save_checkpoint(self, path: str) -> None:
        """Save T5 model + tokenizer + custom module weights to ``path``."""
        import os
        try:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            torch.save(
                {
                    "section_attention": self._section_attention.state_dict(),
                    "temporal_embedding": self._temporal_embedding.state_dict(),
                    "encoder_out_proj": self._encoder_out_proj.state_dict(),
                },
                os.path.join(path, "ssd_extra_modules.pt"),
            )
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to save SSD checkpoint to '{path}': {exc}"
            ) from exc

    def load_checkpoint(self, path: str) -> None:
        """Load T5 model + tokenizer + custom module weights from ``path``."""
        import os
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            # PEFT-format checkpoint (saved after apply_lora): adapter_config.json present.
            # Must load base model first, then attach the saved LoRA adapters.
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                from peft import PeftModel
                base = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
                self._model = PeftModel.from_pretrained(base, path)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)
            extras = torch.load(
                os.path.join(path, "ssd_extra_modules.pt"),
                map_location=self._device,
            )
            self._section_attention.load_state_dict(extras["section_attention"])
            self._temporal_embedding.load_state_dict(extras["temporal_embedding"])
            self._encoder_out_proj.load_state_dict(extras["encoder_out_proj"])
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to load SSD checkpoint from '{path}': {exc}"
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_step_texts(self, trajectory: TrajectoryModel) -> list[str]:
        """Convert TrajectoryModel steps to a windowed list of step text strings."""
        import json as _json

        step_texts: list[str] = []
        for step in trajectory.steps:
            parts = [f"[Step {step.step_index}]", f"Thought: {step.thought}"]
            if step.action is not None:
                parts.append(
                    f"Action: {step.action.tool_name}({_json.dumps(step.action.arguments)})"
                )
            if step.observation is not None:
                obs = step.observation
                parts.append(f"Observation: {obs.content[:300]}")
            step_texts.append("\n".join(parts))

        return _apply_step_window(step_texts)

    def _encode_steps(self, step_texts: list[str]) -> torch.Tensor:
        """
        Encode each step independently via the T5 encoder.

        Returns [N, H] tensor with temporal position embeddings added.
        """
        if not step_texts:
            hidden_dim = self._model.config.d_model
            return torch.zeros(1, hidden_dim, device=self._device)

        step_embs: list[torch.Tensor] = []
        for idx, text in enumerate(step_texts):
            input_ids = self._tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self._max_step_tokens,
                truncation=True,
            ).to(self._device)

            enc_out = self._model.encoder(input_ids=input_ids)
            # Mean-pool over token dimension → [H]
            step_emb = enc_out.last_hidden_state.squeeze(0).mean(dim=0)
            step_embs.append(step_emb)

        embs = torch.stack(step_embs, dim=0)   # [N, H]

        # Add temporal position embeddings (clamped to _MAX_STEPS - 1)
        indices = torch.arange(len(step_texts), device=self._device).clamp(max=_MAX_STEPS - 1)
        pos_embs = self._temporal_embedding(indices)  # [N, H]
        return embs + pos_embs

    def _encode_single_text(self, text: str) -> torch.Tensor:
        """Encode a short text string → [1, H] tensor (mean-pooled encoder output)."""
        input_ids = self._tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=64,
            truncation=True,
        ).to(self._device)
        enc_out = self._model.encoder(input_ids=input_ids)
        emb = enc_out.last_hidden_state.squeeze(0).mean(dim=0)  # [H]
        return emb.unsqueeze(0)                                  # [1, H]

    def _pack_encoder_output(
        self,
        section_contexts: torch.Tensor,  # [5, H]
        ledger_emb: torch.Tensor,        # [1, H]
    ) -> torch.Tensor:                   # [1, 6, H]
        """
        Pack section contexts and ledger embedding into a [1, 6, H] tensor
        to serve as the T5 decoder's encoder output.

        The 6 positions correspond to:
          0 → HARD_CONSTRAINT_LEDGER (ledger embedding)
          1 → SOFT_CONSTRAINTS_SUMMARY
          2 → DECISIONS_MADE
          3 → OPEN_QUESTIONS
          4 → KEY_DISCOVERIES
          5 → CURRENT_ITINERARY_SKETCH

        The decoder's cross-attention will learn which encoder position to
        attend to when generating each section's content.
        """
        packed = torch.cat([ledger_emb, section_contexts], dim=0)  # [6, H]
        packed = self._encoder_out_proj(packed)                    # [6, H] (projected)
        return packed.unsqueeze(0)                                 # [1, 6, H]

    def _build_ledger_text(self, previous_state: CompressedState | None) -> str:
        """Build a short ledger context string from the previous state (if any)."""
        if previous_state is None:
            return "HARD_CONSTRAINT_LEDGER: initialising"
        ledger = previous_state.hard_constraint_ledger
        return (
            f"HARD_CONSTRAINT_LEDGER: {len(ledger.satisfied_ids)} satisfied, "
            f"{len(ledger.violated_ids)} violated, "
            f"{len(ledger.unknown_ids)} unknown"
        )

    def _get_generation_prefix_ids(self) -> torch.Tensor:
        """
        Token IDs for the generation prefix injected at decoder start.
        Forces the decoder to begin generating section markers immediately.
        """
        return self._tokenizer.encode(
            _TEMPLATE_GENERATION_PREFIX,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)

    def _parse_or_fallback(
        self,
        generated_text: str,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None,
    ) -> CompressedState:
        """Parse generated template text; fall back to minimal state on failure."""
        try:
            state = self._template.parse(
                text=generated_text,
                trajectory_id=trajectory.trajectory_id,
                step_index=trajectory.total_steps,
                compression_method=self._METHOD,
            )
        except Exception:
            prior_ledger = (
                previous_state.hard_constraint_ledger
                if previous_state
                else HardConstraintLedger(
                    constraints=(), satisfied_ids=(), violated_ids=(), unknown_ids=()
                )
            )
            state = CompressedState(
                state_id=str(uuid.uuid4()),
                trajectory_id=trajectory.trajectory_id,
                step_index=trajectory.total_steps,
                hard_constraint_ledger=prior_ledger,
                soft_constraints_summary="(parse failed — generation did not match template)",
                decisions_made=[],
                open_questions=[],
                key_discoveries=[],
                current_itinerary_sketch=__import__("re").sub(r"##\s+\w+\s+##", "", (generated_text or "")[:400].strip()).strip() or "(pending)",
                compression_method=self._METHOD,
                token_count=None,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        return state

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _split_trajectory_text(trajectory_text: str) -> list[str]:
    """
    Split a full trajectory text string into per-step strings.

    Splits on ``[Step N]`` headers produced by ``TrajectoryModel.to_text()``.
    Returns a list of step text strings, each starting with its header.
    """
    parts = _STEP_HEADER_RE.split(trajectory_text)
    return [p.strip() for p in parts if p.strip()]


def _apply_step_window(step_texts: list[str]) -> list[str]:
    """
    Apply a sliding window to keep a bounded number of steps.

    Retains the first _N_ANCHOR anchor steps (establish trip context)
    plus the most recent steps up to _MAX_STEPS total.
    """
    if len(step_texts) <= _MAX_STEPS:
        return step_texts
    anchors = step_texts[:_N_ANCHOR]
    recent = step_texts[-(max(_MAX_STEPS - _N_ANCHOR, 1)):]
    # Deduplicate in case anchor and recent overlap (short trajectories)
    seen: set[str] = set()
    result: list[str] = []
    for s in anchors + recent:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return result


def _extract_ledger_text_from_template(compressed_text: str) -> str:
    """
    Extract the HARD_CONSTRAINT_LEDGER section text from a rendered template.

    Used in ``get_log_probs()`` to reconstruct the ledger embedding consistently.
    Falls back to a placeholder if the section is not found.
    """
    pattern = re.compile(
        r"## HARD_CONSTRAINT_LEDGER ##\s*(.*?)(?=## \w+ ##|$)", re.DOTALL
    )
    match = pattern.search(compressed_text)
    if match:
        return match.group(1).strip()[:256]
    return "HARD_CONSTRAINT_LEDGER: unknown"
