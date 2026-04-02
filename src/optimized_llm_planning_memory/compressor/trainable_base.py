"""
compressor/trainable_base.py
============================
TrainableCompressorBase — extends CompressorBase with the RL training contract.

This ABC is the type that ``CompressorPolicy`` (SB3 policy wrapper) and
``RLTrainer`` accept. Non-trainable compressors (LLMCompressor) should NOT
inherit from this class.

Design: LoRA + layer freezing as hooks
----------------------------------------
``apply_lora()`` and ``freeze_base_layers()`` are concrete methods provided
here rather than in the concrete subclass. They delegate to ``lora_utils.py``
so that the LoRA injection logic is centralised and not duplicated across
TransformerCompressor and any future trainable architectures.

Note on LoRA
------------
Even if ``use_lora=False`` in config (the default), the method is still
available. This lets researchers enable LoRA mid-experiment by calling
``apply_lora()`` on an existing checkpoint without rewriting the class.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn

from optimized_llm_planning_memory.compressor.base import CompressorBase
from optimized_llm_planning_memory.core.config import LoRAConfig
from optimized_llm_planning_memory.core.exceptions import CompressorCheckpointError
from optimized_llm_planning_memory.core.models import CompressedState, TrajectoryModel


class TrainableCompressorBase(CompressorBase):
    """
    Extends CompressorBase with the full RL training contract.

    Subclasses must implement
    -------------------------
    - ``compress()``                — from CompressorBase
    - ``get_log_probs()``           — token log-probs for PPO
    - ``get_trainable_parameters()``— params to optimise
    - ``save_checkpoint()``         — serialise model weights
    - ``load_checkpoint()``         — deserialise model weights
    """

    # ── RL training contract ──────────────────────────────────────────────────

    @abstractmethod
    def get_log_probs(
        self,
        trajectory_text: str,
        compressed_text: str,
    ) -> torch.Tensor:
        """
        Compute per-token log p(compressed_token | trajectory_context).

        Shape: (sequence_length,)

        This is used by ``CompressorPolicy.evaluate_actions()`` to compute
        the PPO policy ratio r_t(θ) = exp(log_π_θ(a|s) - log_π_old(a|s)).
        """

    @abstractmethod
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """
        Return parameters that receive gradient updates during PPO.

        When LoRA is active: return only LoRA adapter parameters.
        When base layers are frozen: return only unfrozen parameters.
        """

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """
        Serialise model weights (and optional LoRA adapters) to ``path``.

        Should save enough state to fully reconstruct the compressor for
        inference and continued training.
        """

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load model weights from ``path``.

        Raises
        ------
        CompressorCheckpointError
            If the file is missing, corrupted, or incompatible.
        """

    # ── LoRA + layer freezing (concrete) ─────────────────────────────────────

    def apply_lora(self, lora_config: LoRAConfig) -> None:
        """
        Inject LoRA adapters into the model's attention layers.

        Delegates to ``lora_utils.inject_lora()``. The model must have a
        ``self._model`` attribute pointing to the underlying ``nn.Module``.

        After calling this, ``get_trainable_parameters()`` should return
        only the LoRA adapter parameters.
        """
        from optimized_llm_planning_memory.compressor.lora_utils import inject_lora

        model = self._get_nn_module()
        inject_lora(model, lora_config)

    def freeze_base_layers(self, freeze: bool = True) -> None:
        """
        Freeze (or unfreeze) all base model parameters.

        When freeze=True: only LoRA adapter parameters (if applied) remain
        trainable. When freeze=False: all parameters are trainable.

        Delegates to ``lora_utils.freeze_base_layers()``.
        """
        from optimized_llm_planning_memory.compressor.lora_utils import freeze_base_layers

        model = self._get_nn_module()
        freeze_base_layers(model, freeze=freeze)

    def _get_nn_module(self) -> nn.Module:
        """
        Return the underlying ``nn.Module``.

        Subclasses must expose their model via ``self._model``.
        Raises ``AttributeError`` if not set.
        """
        if not hasattr(self, "_model"):
            raise AttributeError(
                f"{type(self).__name__} must set ``self._model`` (an nn.Module) "
                f"before calling apply_lora() or freeze_base_layers()."
            )
        return self._model  # type: ignore[attr-defined]
