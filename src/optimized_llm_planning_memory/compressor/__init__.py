"""
compressor/
===========
Context compression mechanism — the trained policy in our RL framework.

Design: Two-level ABC hierarchy
---------------------------------
``CompressorBase`` (ABC)
    The universal interface. All compressors — trainable or not — implement
    ``compress()``. Non-trainable compressors (LLMCompressor) inherit from
    this directly. ``get_log_probs()`` raises ``LogProbsNotSupportedError``
    by default, making it clear that not all compressors are RL-trainable.

``TrainableCompressorBase(CompressorBase)`` (ABC)
    Extends the base with the RL training contract: ``get_log_probs()``,
    ``get_trainable_parameters()``, ``save/load_checkpoint()``, LoRA + freeze
    utilities. Only compressors that inherit from THIS class can be used as
    the policy in the PPO training loop.

Why two levels instead of one?
    ``get_log_probs()`` only makes sense for models with explicit token
    distributions (HuggingFace seq2seq/decoder). An LLM API compressor has no
    access to token log-probs. Putting the method at the base level would force
    LLMCompressor to raise NotImplementedError, which is misleading. The
    two-level hierarchy makes the contract explicit at the type level.

Contents
--------
base.py                    — CompressorBase ABC
trainable_base.py          — TrainableCompressorBase ABC
llm_compressor.py          — LLMCompressor (litellm + instructor; not trainable)
transformer_compressor.py  — TransformerCompressor (HF model; T5/BART/decoder)
hybrid_compressor.py       — HybridCompressor (slot extraction + free-form)
template.py                — CompressedStateTemplate (section definitions + renderer)
lora_utils.py              — inject_lora(), freeze_base_layers() helpers
"""
