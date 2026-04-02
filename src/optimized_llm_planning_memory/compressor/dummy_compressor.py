"""
compressor/dummy_compressor.py
================================
DummyCompressor — a minimal seq2seq transformer built from PyTorch primitives.

Purpose
-------
This compressor exists as a concrete, trainable reference implementation.
It demonstrates that the ``TrainableCompressorBase`` interface is fully wired
up end-to-end without requiring a HuggingFace model download. The outputs are
gibberish until the model is trained (which is expected — it is randomly
initialised).

Architecture: from-scratch seq2seq transformer
-----------------------------------------------
All components are written using PyTorch primitives (``nn.Module``,
``nn.Embedding``, ``nn.TransformerEncoderLayer``, etc.) without HuggingFace.

  Tokenizer  : Character-level. Vocab = 128 ASCII chars + PAD/BOS/EOS = 131 tokens.
  Encoder    : Embedding + learned positions → N × TransformerEncoderLayer
  Decoder    : Embedding + learned positions → N × TransformerDecoderLayer (causal)
  Output head: Linear(d_model, vocab_size) — weight-tied to embedding

This is intentionally small (d_model=64, 2 layers by default) so it runs on CPU
in unit tests with no GPU.

Training signal
---------------
``get_log_probs(trajectory_text, compressed_text)`` runs a teacher-forcing
forward pass and returns per-token log-probabilities. These are consumed by
``CompressorPolicy.evaluate_actions()`` to compute the PPO clipping ratio.
The returned tensor requires gradient, so PPO can backpropagate through it.

Weight tying
------------
The output projection shares weights with the embedding matrix. This halves
the parameter count and is standard practice in language modelling.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)

# ── Vocabulary ─────────────────────────────────────────────────────────────────

PAD_ID: int = 0   # padding token
BOS_ID: int = 1   # beginning-of-sequence
EOS_ID: int = 2   # end-of-sequence

# ASCII chars 0–127 are mapped to IDs 3–130 (offset by 3 to reserve PAD/BOS/EOS)
_ASCII_OFFSET = 3
VOCAB_SIZE: int = 128 + _ASCII_OFFSET  # 131


def char_to_id(c: str) -> int:
    """Map a single character to its vocabulary ID."""
    code = ord(c)
    if 0 <= code < 128:
        return code + _ASCII_OFFSET
    return PAD_ID  # non-ASCII → pad


def id_to_char(token_id: int) -> str:
    """Map a vocabulary ID back to a character. Returns '' for special tokens."""
    if token_id < _ASCII_OFFSET:
        return ""
    code = token_id - _ASCII_OFFSET
    if 0 <= code < 128:
        return chr(code)
    return ""


def encode_text(text: str, max_len: int) -> list[int]:
    """
    Encode text to a token ID list with BOS/EOS, truncated to ``max_len``.

    Layout: [BOS, char_1, char_2, ..., EOS]  (at most max_len tokens total)
    """
    char_ids = [char_to_id(c) for c in text]
    # Reserve 1 slot for BOS and 1 for EOS
    char_ids = char_ids[: max_len - 2]
    return [BOS_ID] + char_ids + [EOS_ID]


def decode_ids(token_ids: list[int]) -> str:
    """Decode a list of token IDs to a string (special tokens are skipped)."""
    return "".join(id_to_char(i) for i in token_ids)


def _pad_to(token_ids: list[int], length: int) -> list[int]:
    """Right-pad a token ID list to ``length`` with PAD_ID."""
    n = len(token_ids)
    if n >= length:
        return token_ids[:length]
    return token_ids + [PAD_ID] * (length - n)


# ── Model ──────────────────────────────────────────────────────────────────────

class _DummyTransformerModel(nn.Module):
    """
    Minimal seq2seq transformer.

    All implementation uses standard PyTorch nn building blocks; no HuggingFace.

    Parameters
    ----------
    vocab_size         : Size of the shared token vocabulary.
    d_model            : Embedding / hidden dimension.
    nhead              : Number of attention heads.  Must divide d_model.
    num_encoder_layers : Number of TransformerEncoderLayer stacks.
    num_decoder_layers : Number of TransformerDecoderLayer stacks.
    dim_feedforward    : Feed-forward hidden dimension inside each layer.
    max_len            : Maximum sequence length (for positional embedding).
    dropout            : Dropout rate applied in encoder/decoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        max_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # ── Token embedding (shared between encoder input, decoder input, output head)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)

        # ── Learned positional encoding (simpler than sinusoidal; trainable)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # ── Encoder stack
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # ── Decoder stack
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # ── Output projection — weight-tied to embedding
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for linear layers; normal for embeddings."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2 and p is not self.embedding.weight:
                nn.init.xavier_uniform_(p)

    def _positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return position indices clamped to [0, max_len-1]."""
        positions = torch.arange(
            min(seq_len, self.max_len), device=device
        ).unsqueeze(0)  # (1, T)
        return positions

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode source token IDs to memory.

        Parameters
        ----------
        src_ids : (B, S) int64 tensor

        Returns
        -------
        memory : (B, S, d_model) float tensor
        """
        B, S = src_ids.shape
        S = min(S, self.max_len)
        src_ids = src_ids[:, :S]

        src_pad_mask = src_ids == PAD_ID  # (B, S) — True where padded
        tok_emb = self.embedding(src_ids)  # (B, S, d_model)
        pos_emb = self.pos_embedding(self._positions(S, src_ids.device))  # (1, S, d_model)
        x = tok_emb + pos_emb

        memory = self.encoder(x, src_key_padding_mask=src_pad_mask)  # (B, S, d_model)
        return memory

    def decode_prefix(
        self, tgt_ids: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode target prefix with causal masking (teacher-forcing or autoregressive).

        Parameters
        ----------
        tgt_ids : (B, T) int64 tensor
        memory  : (B, S, d_model) float tensor from encode()

        Returns
        -------
        logits : (B, T, vocab_size) float tensor
        """
        B, T = tgt_ids.shape
        T = min(T, self.max_len)
        tgt_ids = tgt_ids[:, :T]

        # Causal mask: position i can only attend to positions ≤ i
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt_ids.device
        )  # (T, T)

        tok_emb = self.embedding(tgt_ids)  # (B, T, d_model)
        pos_emb = self.pos_embedding(self._positions(T, tgt_ids.device))  # (1, T, d_model)
        x = tok_emb + pos_emb

        out = self.decoder(x, memory, tgt_mask=tgt_mask)  # (B, T, d_model)
        logits = self.output_projection(out)               # (B, T, vocab_size)
        return logits

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forcing forward pass for training.

        Parameters
        ----------
        src_ids : (B, S) int64 source token IDs
        tgt_ids : (B, T) int64 target token IDs (shifted right — starts with BOS)

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        memory = self.encode(src_ids)
        return self.decode_prefix(tgt_ids, memory)

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, max_output_len: int) -> list[int]:
        """
        Greedy autoregressive decoding for the first element of the batch.

        Parameters
        ----------
        src_ids       : (B, S) int64 — uses only index 0
        max_output_len: Maximum number of output tokens to generate.

        Returns
        -------
        list[int] — generated token IDs (includes BOS, may include EOS)
        """
        memory = self.encode(src_ids[:1])  # (1, S, d_model) — first example only
        generated: list[int] = [BOS_ID]

        for _ in range(max_output_len - 1):
            tgt_so_far = torch.tensor(
                [generated], dtype=torch.long, device=src_ids.device
            )
            logits = self.decode_prefix(tgt_so_far, memory)  # (1, T, vocab_size)
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if next_id == EOS_ID:
                break

        return generated


# ── DummyCompressor ────────────────────────────────────────────────────────────

class DummyCompressor(TrainableCompressorBase):
    """
    Concrete TrainableCompressorBase built on ``_DummyTransformerModel``.

    This compressor is deliberately small (d_model=64, 2 layers) so it can run
    on CPU in tests. Outputs are meaningless until fine-tuned, which is expected.

    Use this to:
    - Verify the full RL training loop works end-to-end.
    - Benchmark against learned compression without a GPU.
    - Prototype reward shaping before training a larger HF model.

    Parameters
    ----------
    d_model             : Hidden dimension (default 64).
    nhead               : Number of attention heads (default 4; must divide d_model).
    num_encoder_layers  : Number of encoder layers (default 2).
    num_decoder_layers  : Number of decoder layers (default 2).
    dim_feedforward     : FF hidden size (default 128).
    max_input_len       : Max tokens the encoder will accept (default 512).
    max_output_len      : Max tokens the decoder will generate (default 128).
    device              : 'cpu' | 'cuda' | 'auto'.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        max_input_len: int = 512,
        max_output_len: int = 128,
        device: str = "cpu",
    ) -> None:
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        self._max_input_len = max_input_len
        self._max_output_len = max_output_len

        # Resolve device
        if device == "auto":
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = device
        self._device = torch.device(_device)

        # Build the underlying nn.Module (exposed as self._model for apply_lora etc.)
        self._model = _DummyTransformerModel(
            vocab_size=VOCAB_SIZE,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_len=max(max_input_len, max_output_len),
        ).to(self._device)
        self._model.eval()

    # ── CompressorBase interface ───────────────────────────────────────────────

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Generate a CompressedState from a trajectory.

        The text output is produced by greedy decoding and will be gibberish
        until the model is trained. The structure (all required fields) is
        always valid.

        Parameters
        ----------
        trajectory     : Frozen trajectory model.
        previous_state : Prior CompressedState (used as prefix hint if provided).

        Returns
        -------
        CompressedState with all required fields populated.
        """
        # Encode trajectory text
        traj_text = trajectory.to_text() if hasattr(trajectory, "to_text") else _trajectory_to_text(trajectory)

        # Optionally prepend prior state sketch as context
        if previous_state is not None:
            traj_text = previous_state.current_itinerary_sketch + "\n" + traj_text

        src_ids = self._encode_input(traj_text)

        # Greedy decode
        self._model.eval()
        with torch.no_grad():
            generated_ids = self._model.greedy_decode(src_ids, self._max_output_len)

        generated_text = decode_ids(generated_ids).strip()

        # Fall back to a placeholder if nothing was generated
        if not generated_text:
            generated_text = f"[step {trajectory.total_steps}] planning in progress"

        # Build a minimal valid CompressedState
        # The hard constraint ledger mirrors the trajectory ID
        ledger = HardConstraintLedger(
            constraints=(),
            satisfied_ids=(),
            violated_ids=(),
            unknown_ids=(),
        )

        # Use the generated text as the itinerary sketch
        # soft_constraints_summary must be non-empty for template.validate()
        soft_summary = generated_text[:80] if generated_text else "no summary available"

        return CompressedState(
            state_id=str(uuid.uuid4()),
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            hard_constraint_ledger=ledger,
            soft_constraints_summary=soft_summary,
            decisions_made=list(
                f"step {s.step_index}: {s.action.tool_name}" if s.action else f"step {s.step_index}: no action"
                for s in trajectory.steps[-3:]
            ),
            open_questions=["further planning needed"],
            key_discoveries=[generated_text[:60]] if generated_text else [],
            current_itinerary_sketch=generated_text,
            compression_method="dummy",
            token_count=len(generated_ids),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    # ── TrainableCompressorBase interface ─────────────────────────────────────

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Compute per-token log p(compressed_token | trajectory_context).

        Uses teacher-forcing: given the source and the full target, compute
        the log probability assigned to each target token by the model.

        The returned tensor has ``requires_grad=True`` so PPO can backpropagate
        through it.

        Parameters
        ----------
        trajectory_text  : Source text (encoder input).
        compressed_text  : Target text produced at rollout time (decoder target).

        Returns
        -------
        log_probs : (T,) float tensor  where T = len(target tokens) - 1
                    Per-token log p(t_i | context, t_1, ..., t_{i-1}).
        """
        src_ids = self._encode_input(trajectory_text)         # (1, S)
        # Encode target WITHOUT padding so T == actual token count (BOS + chars + EOS)
        raw_ids = encode_text(compressed_text, self._max_output_len)
        tgt_ids = torch.tensor([raw_ids], dtype=torch.long, device=self._device)  # (1, T)

        # Shift target: input  = tgt[:-1] (BOS ... last-1)
        #               labels = tgt[1:]  (first-1 ... EOS)
        if tgt_ids.shape[1] < 2:
            # Degenerate case: target is too short to compute log probs
            return torch.zeros(1, device=self._device, requires_grad=True)

        tgt_in = tgt_ids[:, :-1]   # (1, T-1)
        tgt_out = tgt_ids[:, 1:]   # (1, T-1) — ground truth labels

        # Forward pass (training mode for gradient support)
        self._model.train()
        logits = self._model(src_ids, tgt_in)  # (1, T-1, vocab_size)

        # Log softmax → per-token log probs
        log_probs_all = F.log_softmax(logits, dim=-1)  # (1, T-1, vocab_size)

        # Gather the log probs of the actual target tokens
        # tgt_out: (1, T-1) → (1, T-1, 1) for gather
        gathered = log_probs_all.gather(
            dim=2, index=tgt_out.unsqueeze(-1)
        )  # (1, T-1, 1)

        return gathered.squeeze(-1).squeeze(0)  # (T-1,)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all model parameters that should receive gradient updates."""
        return [p for p in self._model.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str) -> None:
        """
        Save model weights to ``{path}/dummy_compressor.pt``.

        Creates the directory if it does not exist.
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self._model.state_dict(),
            "config": {
                "d_model": self._model.d_model,
                "vocab_size": VOCAB_SIZE,
                "max_input_len": self._max_input_len,
                "max_output_len": self._max_output_len,
            },
        }
        torch.save(ckpt, save_dir / "dummy_compressor.pt")

    def load_checkpoint(self, path: str) -> None:
        """
        Load model weights from ``{path}/dummy_compressor.pt`` or ``path`` directly.

        Raises
        ------
        CompressorCheckpointError
            If the file does not exist or the state dict is incompatible.
        """
        p = Path(path)
        if p.is_dir():
            p = p / "dummy_compressor.pt"
        if not p.exists():
            raise CompressorCheckpointError(
                f"DummyCompressor checkpoint not found at {p}."
            )
        try:
            ckpt = torch.load(p, map_location=self._device)
            self._model.load_state_dict(ckpt["state_dict"])
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to load DummyCompressor checkpoint from {p}: {exc}"
            ) from exc

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode_input(self, text: str) -> torch.Tensor:
        """Tokenize and tensorise source text → (1, S)."""
        ids = encode_text(text, self._max_input_len)
        ids = _pad_to(ids, self._max_input_len)
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def _encode_target(self, text: str) -> torch.Tensor:
        """Tokenize and tensorise target text → (1, T)."""
        ids = encode_text(text, self._max_output_len)
        ids = _pad_to(ids, self._max_output_len)
        return torch.tensor([ids], dtype=torch.long, device=self._device)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _trajectory_to_text(trajectory: TrajectoryModel) -> str:
    """Fallback text representation for TrajectoryModel (no to_text() method)."""
    lines: list[str] = [f"Trajectory {trajectory.trajectory_id} steps={trajectory.total_steps}"]
    for step in trajectory.steps:
        lines.append(f"[step {step.step_index}] thought={step.thought[:80]}")
        if step.action:
            lines.append(f"  action={step.action.tool_name}")
    return "\n".join(lines)
