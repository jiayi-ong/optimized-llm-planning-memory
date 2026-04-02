"""
compressor/lora_utils.py
========================
Utilities for LoRA adapter injection and layer freezing.

These functions are called by ``TrainableCompressorBase.apply_lora()``
and ``TrainableCompressorBase.freeze_base_layers()``. They are kept in a
separate module so that:
  1. The LoRA injection logic is centralised and not duplicated.
  2. Future compressor architectures can reuse these utilities without
     inheriting from TransformerCompressor.

Implementation uses HuggingFace PEFT (Parameter-Efficient Fine-Tuning).
"""

from __future__ import annotations

import logging

import torch.nn as nn

from optimized_llm_planning_memory.core.config import LoRAConfig

logger = logging.getLogger(__name__)


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Inject LoRA adapters into the model using HuggingFace PEFT.

    After calling this function:
    - Base model weights are frozen.
    - Only LoRA adapter parameters are trainable.
    - ``model.print_trainable_parameters()`` (PEFT method) shows the breakdown.

    Parameters
    ----------
    model  : The ``nn.Module`` to inject LoRA into (e.g., a HuggingFace model).
    config : LoRA hyperparameters (r, alpha, dropout, target_modules).

    Returns
    -------
    nn.Module
        The model wrapped with PEFT LoRA (a ``PeftModel``).

    Raises
    ------
    ImportError
        If the ``peft`` package is not installed.
    """
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise ImportError(
            "The 'peft' package is required for LoRA. Install it with: pip install peft"
        ) from exc

    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # adjust for decoder-only: TaskType.CAUSAL_LM
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    trainable, total = _count_parameters(peft_model)
    logger.info(
        "LoRA applied: %d trainable params / %d total (%.2f%%)",
        trainable, total, 100.0 * trainable / total if total > 0 else 0.0,
    )
    return peft_model


def freeze_base_layers(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze all base model parameters.

    When ``freeze=True``: sets ``requires_grad=False`` on all parameters
    that are NOT part of a LoRA adapter (i.e., whose name does not contain
    'lora_'). This allows LoRA adapters to remain trainable.

    When ``freeze=False``: unfreezes ALL parameters (for full fine-tuning).

    Parameters
    ----------
    model  : The model to freeze/unfreeze.
    freeze : True to freeze base layers; False to unfreeze all.
    """
    for name, param in model.named_parameters():
        if freeze:
            # Keep LoRA adapter params trainable; freeze everything else
            if "lora_" not in name:
                param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(
        "Layer freeze=%s: %d frozen params, %d trainable params.",
        freeze, frozen, trainable,
    )


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
