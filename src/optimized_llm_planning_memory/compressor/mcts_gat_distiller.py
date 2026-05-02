"""
compressor/mcts_gat_distiller.py
=================================
MCTSGraphAttentionDistiller (TGAD) — Design 2: MCTS-aware trainable compressor.

Design: Tree-conditioned distillation via path-set attention
------------------------------------------------------------
The current ``LLMMCTSCompressor`` distils the MCTS tree by dumping path texts
into an LLM prompt. This has three problems:

  1. Token economy: distillation consumes the same rate-limited model budget
     as the planner.
  2. Untrainability: the LLM is frozen; no RL signal shapes its selection policy.
  3. Verified/speculative blindness: the LLM cannot distinguish real tool
     observations from synthetic MCTS expansion steps (observation=None).

TGAD replaces the LLM distillation call with a neural pipeline:

    1. Materialise paths from MCTSTreeRepresentation (best + alt paths).
    2. Encode each path via FROZEN T5 encoder (mean-pool → [H]).
    3. Project structural features [q_value, is_verified, is_best_path, ...]
       → [64] via small MLP and concatenate with path embedding.
    4. Run 2-layer PathSetEncoder (from ``tree_gat.py``) over N path nodes.
    5. Score each path with Linear → sigmoid (Importance Scorer).
       Apply Gumbel-softmax top-K during training (differentiable);
       hard top-K during inference.
    6. Attention-weighted pool selected paths → single context vector [H].
    7. T5 decoder + LoRA generates all 6 template sections + top_candidates
       and tradeoffs from this context.

Key innovations
---------------
  - Verified/speculative distinction is explicit: ``is_verified`` is a
    binary feature computed from whether a path's steps have real observations.
    The PathSetEncoder learns to down-weight speculative branches.
  - Zero LLM calls at distillation: fully neural, ~10 ms on T4.
  - During training: num_simulations can be reduced to 10 (vs. eval-time 50)
    because the trained distiller extracts richer signal per simulation.

RL training curriculum (recommended)
--------------------------------------
  Phase 1 — Supervised pre-training (~1 hr, no PPO):
    Run LLMMCTSCompressor on 200 episodes offline to generate
    (MCTSTreeRepresentation, CompressedState) pairs. Train TGAD via cross-
    entropy on CompressedState token sequences. Warms up GAT + decoder.

  Phase 2 — PPO fine-tuning (~4 hr, reduced MCTS):
    PPO with num_simulations=10, ppo_colab.yaml settings.

Fallback (no MCTS)
------------------
``compress()`` is implemented as a fallback: when no MCTSTreeRepresentation
is available, the best-path trajectory is treated as the single "path node"
and the rest of the pipeline runs identically. This enables ablation studies
without changing the agent mode or config.

Memory (Colab T4)
-----------------
  T5-small encoder frozen (fp16): ~80 MB
  T5-small decoder + LoRA (fp16/fp32): ~86 MB
  GAT (PathSetEncoder, fp32): ~3 MB
  Feature projector + importance scorer (fp32): ~0.5 MB
  Adam optimizer state: ~18 MB
  Training activations: ~250 MB
  ──────────────────────────────────────────
  Total: ~437 MB — well within T4's 16 GB.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from optimized_llm_planning_memory.compressor.mcts_aware import MCTSAwareCompressor
from optimized_llm_planning_memory.compressor.template import CompressedStateTemplate
from optimized_llm_planning_memory.compressor.trainable_base import TrainableCompressorBase
from optimized_llm_planning_memory.compressor.tree_gat import PathSetEncoder
from optimized_llm_planning_memory.core.config import LoRAConfig
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import (
    CompressedState,
    HardConstraintLedger,
    TrajectoryModel,
)

if TYPE_CHECKING:
    from optimized_llm_planning_memory.mcts.node import MCTSTreeRepresentation

# Number of structural scalars per path node.
# [q_value, is_verified, is_best_path, norm_path_idx, has_alt_bookings]
_STRUCT_FEATURE_DIM = 5

# Max alternative paths to materialise from MCTSTreeRepresentation.
# Keeps node count bounded: 1 best + MAX_ALT_PATHS alts = MAX_NODES nodes.
_MAX_ALT_PATHS = 3
_MAX_NODES = 1 + _MAX_ALT_PATHS   # = 4

# Gumbel-softmax temperature schedule
_GUMBEL_TAU_TRAIN = 1.0   # exploration during training
_GUMBEL_TAU_EVAL = 0.01   # near-hard selection during inference

# Template suffix appended to the generation prefix to guide top_candidates output
_TEMPLATE_GENERATION_PREFIX = "## HARD_CONSTRAINT_LEDGER ##\n"

# Max tokens used to encode each path's trajectory text
_MAX_PATH_TOKENS = 256


class StructuralFeatureProjector(nn.Module):
    """
    Projects per-path scalar features → [proj_dim] and concatenates with
    the path text embedding to form a unified node feature vector.

    Input features (5 scalars per path):
      0 — q_value        : MCTS Q-value (float, [0, 1])
      1 — is_verified    : 1 if path has only real observations, 0 if any synthetic
      2 — is_best_path   : 1 for the best-path node, 0 for alternatives
      3 — norm_path_idx  : normalised path index (0 = best, 1/(N-1) … = alts)
      4 — has_bookings   : 1 if path trajectory contains a booking action
    """

    def __init__(self, text_dim: int = 512, struct_dim: int = 5, proj_dim: int = 64) -> None:
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(struct_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self._fuse = nn.Sequential(
            nn.Linear(text_dim + proj_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

    def forward(
        self,
        text_embs: torch.Tensor,        # [N, text_dim]
        struct_feats: torch.Tensor,     # [N, struct_dim]
    ) -> torch.Tensor:                  # [N, text_dim]
        struct_proj = self._mlp(struct_feats)                    # [N, proj_dim]
        combined = torch.cat([text_embs, struct_proj], dim=-1)   # [N, text_dim + proj_dim]
        return self._fuse(combined)                              # [N, text_dim]


class GumbelTopK(nn.Module):
    """
    Differentiable top-K selection via Gumbel-softmax.

    During training (training=True): uses Gumbel-softmax to produce soft
    selection weights. Gradients flow through the selection to the importance
    scorer and the GAT.

    During inference (training=False): uses hard top-K (argmax) for deterministic
    selection.

    Why Gumbel-softmax?
    -------------------
    Standard top-K is non-differentiable (argmax). Gumbel-softmax approximates
    the categorical distribution with a continuous relaxation controlled by
    temperature τ. As τ → 0 the distribution sharpens to a one-hot.

    With τ=1.0 during training, the model receives gradient signal for all
    paths proportionally to their importance scores, enabling the importance
    scorer to learn from RL reward.
    """

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self._k = k

    def forward(
        self,
        importance_logits: torch.Tensor,   # [N]
        node_embs: torch.Tensor,           # [N, H]
    ) -> torch.Tensor:                     # [H] — weighted pooled output
        """
        Returns an attention-weighted pool of the top-K selected nodes.

        During training: soft weights via Gumbel-softmax → differentiable.
        During inference: hard top-K weights (0/1) → deterministic.
        """
        N = importance_logits.size(0)
        k = min(self._k, N)

        if self.training:
            # Gumbel-softmax: add Gumbel noise and take softmax
            tau = _GUMBEL_TAU_TRAIN
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(importance_logits) + 1e-10) + 1e-10
            )
            perturbed = (importance_logits + gumbel_noise) / tau
            weights = F.softmax(perturbed, dim=0)              # [N] — soft selection
        else:
            # Hard top-K during inference
            tau = _GUMBEL_TAU_EVAL
            topk_vals, topk_idx = torch.topk(importance_logits, k)
            weights = torch.zeros(N, device=importance_logits.device)
            weights[topk_idx] = F.softmax(topk_vals / tau, dim=0)

        # Attention-weighted pool → [H]
        return (weights.unsqueeze(-1) * node_embs).sum(dim=0)


class MCTSGraphAttentionDistiller(TrainableCompressorBase, MCTSAwareCompressor):
    """
    Tree-conditioned distiller using path-set attention over MCTS search paths.

    Replaces LLM-based tree distillation (LLMMCTSCompressor) with a trainable
    neural network that explicitly models verified vs. speculative branches and
    cross-path relationships.

    MRO: MCTSGraphAttentionDistiller → TrainableCompressorBase → MCTSAwareCompressor
         → CompressorBase. Python cooperative inheritance handles this cleanly.

    Parameters
    ----------
    model_name_or_path : HuggingFace T5 model ID (default: google/flan-t5-small).
    max_path_tokens    : Max tokens per path's trajectory text for encoding.
    max_output_tokens  : Max tokens in generated compressed state.
    device             : 'cpu' | 'cuda' | 'auto'.
    use_lora           : Apply LoRA to the T5 decoder (encoder stays frozen).
    lora_config        : LoRA hyperparameters.
    top_k_paths        : Number of paths selected by Gumbel top-K.
    """

    _METHOD = "mcts_gat"

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-small",
        max_path_tokens: int = _MAX_PATH_TOKENS,
        max_output_tokens: int = 512,
        device: str = "auto",
        use_lora: bool = True,
        lora_config: LoRAConfig | None = None,
        top_k_paths: int = 3,
    ) -> None:
        self._model_name = model_name_or_path
        self._max_path_tokens = max_path_tokens
        self._max_output_tokens = max_output_tokens
        self._device = self._resolve_device(device)
        self._template = CompressedStateTemplate()
        self._top_k_paths = top_k_paths

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path
        ).to(self._device)

        hidden_dim = self._model.config.d_model   # 512 for T5-small

        # Freeze the encoder entirely (Design 2: encoder runs N times per node,
        # LoRA on encoder would multiply cost N-fold with no benefit here).
        for param in self._model.encoder.parameters():
            param.requires_grad_(False)

        # Trainable modules ---------------------------------------------------

        self._struct_projector = StructuralFeatureProjector(
            text_dim=hidden_dim,
            struct_dim=_STRUCT_FEATURE_DIM,
            proj_dim=64,
        ).to(self._device)

        # PathSetEncoder (from tree_gat.py): 2-layer multi-head path attention
        self._path_encoder = PathSetEncoder(
            dim=hidden_dim, num_heads=4, dropout=0.1
        ).to(self._device)

        # Importance scorer → Gumbel top-K selection
        self._importance_scorer = nn.Linear(hidden_dim, 1).to(self._device)
        self._gumbel_topk = GumbelTopK(k=top_k_paths)

        # Projection that packages [tree_context | best_path_emb] for decoder
        self._context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self._device)

        # Apply LoRA to decoder attention layers only
        if use_lora:
            _cfg = lora_config or LoRAConfig()
            self._apply_decoder_lora(_cfg)

    # ── MCTSAwareCompressor ───────────────────────────────────────────────────

    def compress_with_tree(
        self,
        tree_repr: "MCTSTreeRepresentation",
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Distil an MCTS search tree into a structured CompressedState.

        Does NOT make any LLM calls — fully neural, ~80 ms on T4.

        Pipeline:
          1. Materialise N paths from tree_repr
          2. Encode each path via frozen T5 encoder
          3. Build structural feature vectors (Q-value, verified flag, etc.)
          4. Fuse text + structural features via StructuralFeatureProjector
          5. Run 2-layer PathSetEncoder (cross-path attention)
          6. Score nodes → Gumbel top-K selection → attention pool → context
          7. T5 decoder generates full template + top_candidates + tradeoffs
        """
        nodes = self._materialise_nodes(tree_repr)
        path_embs, struct_feats = self._encode_nodes(nodes)
        node_feats = self._struct_projector(path_embs, struct_feats)
        refined_feats = self._path_encoder(node_feats, n_nodes=len(nodes))
        importance_logits = self._importance_scorer(refined_feats).squeeze(-1)  # [N]
        tree_context = self._gumbel_topk(importance_logits, refined_feats)      # [H]

        # Best-path embedding (index 0 by convention from _materialise_nodes)
        best_path_emb = refined_feats[0]                                        # [H]

        encoder_out = self._pack_encoder_output(tree_context, best_path_emb)   # [1, 2, H]

        prefix_ids = self._get_generation_prefix_ids()
        with torch.no_grad():
            output_ids = self._model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                decoder_input_ids=prefix_ids,
                max_new_tokens=self._max_output_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Recover top_candidates and tradeoffs from tree_repr for the output state
        top_candidates = (
            list(tree_repr.top_candidates[:_MAX_ALT_PATHS])
            if tree_repr.top_candidates else None
        )
        tradeoffs = tree_repr.tradeoffs if tree_repr.tradeoffs else None

        return self._parse_or_fallback(
            generated_text,
            trajectory_id=tree_repr.best_path_trajectory.trajectory_id,
            step_index=tree_repr.best_path_trajectory.total_steps,
            previous_state=previous_state,
            top_candidates=top_candidates,
            tradeoffs=tradeoffs,
        )

    def compress(
        self,
        trajectory: TrajectoryModel,
        previous_state: CompressedState | None = None,
    ) -> CompressedState:
        """
        Fallback compression when no MCTS tree is available.

        Treats the single trajectory as the sole "path node" and runs the
        same encode → attention → decode pipeline. The PathSetEncoder still
        runs (over 1 node), but there are no inter-path contrasts to capture.

        This enables TGAD to be used in ``AgentMode.COMPRESSOR`` for ablation.
        """
        path_text = trajectory.to_text()
        path_emb = self._encode_path_text(path_text).unsqueeze(0)   # [1, H]

        # Structural features: best path, fully verified, Q=1.0 (unknown)
        struct_feats = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self._device,
        )
        node_feats = self._struct_projector(path_emb, struct_feats)  # [1, H]
        refined = self._path_encoder(node_feats, n_nodes=1)          # [1, H]

        importance_logits = self._importance_scorer(refined).squeeze(-1)
        tree_context = self._gumbel_topk(importance_logits, refined)  # [H]
        best_path_emb = refined[0]

        encoder_out = self._pack_encoder_output(tree_context, best_path_emb)

        prefix_ids = self._get_generation_prefix_ids()
        with torch.no_grad():
            output_ids = self._model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
                decoder_input_ids=prefix_ids,
                max_new_tokens=self._max_output_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        generated_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._parse_or_fallback(
            generated_text,
            trajectory_id=trajectory.trajectory_id,
            step_index=trajectory.total_steps,
            previous_state=previous_state,
        )

    # ── TrainableCompressorBase ───────────────────────────────────────────────

    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Compute per-token log p(compressed_token | path_context).

        Uses the fallback path (single trajectory as path node) since
        ``get_log_probs`` receives only a flat trajectory text string,
        not the full MCTSTreeRepresentation. This is consistent with how
        the PPO rollout buffer stores transitions.

        The Phase 1 supervised pre-training uses a different loss (cross-entropy
        on tree-conditioned outputs) computed outside of this method.
        """
        path_emb = self._encode_path_text(trajectory_text).unsqueeze(0)  # [1, H]
        struct_feats = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self._device,
        )
        node_feats = self._struct_projector(path_emb, struct_feats)
        refined = self._path_encoder(node_feats, n_nodes=1)
        importance_logits = self._importance_scorer(refined).squeeze(-1)
        tree_context = self._gumbel_topk(importance_logits, refined)
        best_path_emb = refined[0]

        encoder_out = self._pack_encoder_output(tree_context, best_path_emb)  # [1, 2, H]

        target_ids = self._tokenizer.encode(
            compressed_text,
            return_tensors="pt",
            max_length=self._max_output_tokens,
            truncation=True,
        ).to(self._device)

        bos_id = self._model.config.decoder_start_token_id or 0
        bos = torch.tensor([[bos_id]], dtype=torch.long, device=self._device)
        decoder_input_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)

        outputs = self._model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits.squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)

        target = target_ids.squeeze(0)
        token_log_probs = log_probs.gather(
            dim=-1, index=target.unsqueeze(-1)
        ).squeeze(-1)

        pad_id = self._tokenizer.pad_token_id or 0
        return token_log_probs * (target != pad_id).float()

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Return trainable parameters:
          - LoRA adapters on T5 decoder (encoder is fully frozen)
          - StructuralFeatureProjector parameters
          - PathSetEncoder (BidirectionalPathGAT) parameters
          - Importance scorer
          - Root anchor bias (in PathSetEncoder)
          - Context projection
        """
        params: list[nn.Parameter] = []
        # Decoder LoRA (or full decoder if LoRA not applied)
        params += [p for p in self._model.decoder.parameters() if p.requires_grad]
        # Shared head (lm_head) if trainable
        if hasattr(self._model, "lm_head"):
            params += [p for p in self._model.lm_head.parameters() if p.requires_grad]
        # Our custom trainable modules
        params += list(self._struct_projector.parameters())
        params += list(self._path_encoder.parameters())
        params += list(self._importance_scorer.parameters())
        params += list(self._context_proj.parameters())
        return params

    def save_checkpoint(self, path: str) -> None:
        """Save T5 model + custom modules to ``path``."""
        import os
        try:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            torch.save(
                {
                    "struct_projector": self._struct_projector.state_dict(),
                    "path_encoder": self._path_encoder.state_dict(),
                    "importance_scorer": self._importance_scorer.state_dict(),
                    "context_proj": self._context_proj.state_dict(),
                },
                os.path.join(path, "tgad_extra_modules.pt"),
            )
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to save TGAD checkpoint to '{path}': {exc}"
            ) from exc

    def load_checkpoint(self, path: str) -> None:
        """Load T5 model + custom modules from ``path``."""
        import os
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            # PEFT-format checkpoint (saved after _apply_decoder_lora): adapter_config.json present.
            # Reload the base model from HuggingFace then attach the saved LoRA adapter weights.
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                from peft import PeftModel
                base = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
                self._model = PeftModel.from_pretrained(base, path)
            else:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)
            # Re-freeze encoder regardless of checkpoint format
            for param in self._model.encoder.parameters():
                param.requires_grad_(False)
            extras = torch.load(
                os.path.join(path, "tgad_extra_modules.pt"),
                map_location=self._device,
            )
            self._struct_projector.load_state_dict(extras["struct_projector"])
            self._path_encoder.load_state_dict(extras["path_encoder"])
            self._importance_scorer.load_state_dict(extras["importance_scorer"])
            self._context_proj.load_state_dict(extras["context_proj"])
        except Exception as exc:
            raise CompressorCheckpointError(
                f"Failed to load TGAD checkpoint from '{path}': {exc}"
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _materialise_nodes(
        self,
        tree_repr: "MCTSTreeRepresentation",
    ) -> list[dict]:
        """
        Convert MCTSTreeRepresentation into a list of node dicts.

        Node dict keys:
          text       : trajectory text for encoding
          q_value    : MCTS Q-value (float)
          is_verified: True if all steps in path have real observations
          is_best    : True for the best-path node
          path_idx   : index in the list (0 = best, 1+ = alternatives)
        """
        nodes: list[dict] = []

        # Best path node (always index 0 — PathSetEncoder expects this)
        best_traj = tree_repr.best_path_trajectory
        nodes.append({
            "text": best_traj.to_text()[:1000],       # truncate for encoding
            "q_value": tree_repr.stats.root_value,
            "is_verified": _is_trajectory_verified(best_traj),
            "is_best": True,
            "path_idx": 0,
        })

        # Alternative path nodes (up to _MAX_ALT_PATHS)
        for i, alt_traj in enumerate(tree_repr.alternative_paths[:_MAX_ALT_PATHS]):
            # Look up Q-value: find the last node_id in alt_traj if available
            # We use root_value as fallback since node_values are keyed by UUID
            alt_q = max(tree_repr.node_values.values(), default=0.5) * 0.8  # rough proxy
            nodes.append({
                "text": alt_traj.to_text()[:1000],
                "q_value": alt_q,
                "is_verified": _is_trajectory_verified(alt_traj),
                "is_best": False,
                "path_idx": i + 1,
            })

        return nodes

    def _encode_nodes(
        self,
        nodes: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of node dicts into text embeddings and structural features.

        Returns:
          path_embs   : [N, H] text embeddings (frozen T5 encoder)
          struct_feats: [N, 5] structural feature tensor
        """
        path_embs: list[torch.Tensor] = []
        struct_rows: list[list[float]] = []

        max_idx = max(n["path_idx"] for n in nodes) if nodes else 1

        for node in nodes:
            emb = self._encode_path_text(node["text"])
            path_embs.append(emb)

            struct_rows.append([
                float(node["q_value"]),
                float(node["is_verified"]),
                float(node["is_best"]),
                float(node["path_idx"]) / max(float(max_idx), 1.0),  # norm index
                float(_text_has_bookings(node["text"])),
            ])

        return (
            torch.stack(path_embs, dim=0),                          # [N, H]
            torch.tensor(struct_rows, dtype=torch.float32, device=self._device),  # [N, 5]
        )

    def _encode_path_text(self, text: str) -> torch.Tensor:
        """
        Encode a path's trajectory text via frozen T5 encoder.
        Returns [H] mean-pooled embedding.
        """
        input_ids = self._tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=self._max_path_tokens,
            truncation=True,
        ).to(self._device)

        # Encoder is frozen — no grad needed, saves memory
        with torch.no_grad():
            enc_out = self._model.encoder(input_ids=input_ids)
        return enc_out.last_hidden_state.squeeze(0).mean(dim=0)   # [H]

    def _pack_encoder_output(
        self,
        tree_context: torch.Tensor,    # [H]
        best_path_emb: torch.Tensor,   # [H]
    ) -> torch.Tensor:                 # [1, 2, H]
        """
        Pack tree context + best-path embedding into [1, 2, H] for T5 decoder.

        Position 0 → tree_context (aggregate of selected paths, including alts)
        Position 1 → best_path_emb (the verified best path's representation)

        The decoder cross-attends to both positions, learning:
          pos 0: "what does the full search say?"
          pos 1: "what has the best confirmed path done?"
        """
        packed = torch.stack([
            self._context_proj(tree_context),
            self._context_proj(best_path_emb),
        ], dim=0)                    # [2, H]
        return packed.unsqueeze(0)   # [1, 2, H]

    def _get_generation_prefix_ids(self) -> torch.Tensor:
        return self._tokenizer.encode(
            _TEMPLATE_GENERATION_PREFIX,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)

    def _parse_or_fallback(
        self,
        generated_text: str,
        trajectory_id: str,
        step_index: int,
        previous_state: CompressedState | None,
        top_candidates: list[str] | None = None,
        tradeoffs: str | None = None,
    ) -> CompressedState:
        """Parse generated template text; inject MCTS fields; fall back on failure."""
        try:
            state = self._template.parse(
                text=generated_text,
                trajectory_id=trajectory_id,
                step_index=step_index,
                compression_method=self._METHOD,
            )
            # Inject MCTS-specific fields (CompressedState is frozen, so rebuild)
            if top_candidates is not None or tradeoffs is not None:
                state = CompressedState(
                    **{
                        k: v for k, v in state.model_dump().items()
                        if k not in ("top_candidates", "tradeoffs", "state_id", "created_at")
                    },
                    state_id=str(uuid.uuid4()),
                    created_at=datetime.now(tz=timezone.utc).isoformat(),
                    top_candidates=top_candidates,
                    tradeoffs=tradeoffs,
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
                trajectory_id=trajectory_id,
                step_index=step_index,
                hard_constraint_ledger=prior_ledger,
                soft_constraints_summary="(parse failed — generation did not match template)",
                decisions_made=[],
                open_questions=[],
                key_discoveries=[],
                current_itinerary_sketch=generated_text[:400],
                compression_method=self._METHOD,
                token_count=None,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                top_candidates=top_candidates,
                tradeoffs=tradeoffs,
            )
        return state

    def _apply_decoder_lora(self, lora_config: LoRAConfig) -> None:
        """
        Apply LoRA only to the T5 decoder, leaving the encoder frozen.

        Overrides ``TrainableCompressorBase.apply_lora()`` which would apply
        LoRA to the full model (encoder + decoder). We want the encoder frozen
        because (a) it's used N times for path encoding and (b) the decoder's
        LoRA adapters are sufficient for learning the generation policy.
        """
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ImportError(
                "The 'peft' package is required. Install with: pip install peft"
            ) from exc

        peft_config = PeftLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            target_modules=lora_config.target_modules,
            bias="none",
        )
        # Wrap the full model but only decoder layers will have trainable adapters
        # since encoder is already frozen
        self._model = get_peft_model(self._model, peft_config)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _is_trajectory_verified(trajectory: TrajectoryModel) -> bool:
    """
    Return True if ALL steps in the trajectory have real observations.

    A trajectory is "verified" if every step that produced a tool call
    also has a non-None observation. Trajectories containing any synthetic
    MCTS expansion steps (observation=None) are not verified.
    """
    for step in trajectory.steps:
        if step.action is not None and step.observation is None:
            return False
    return True


def _text_has_bookings(text: str) -> bool:
    """
    Heuristic: check if a path's trajectory text contains booking actions.

    Used as a structural feature to help the model identify high-progress paths.
    """
    booking_keywords = ("book_hotel", "book_event", "select_flight", "booking confirmed")
    text_lower = text.lower()
    return any(kw in text_lower for kw in booking_keywords)
