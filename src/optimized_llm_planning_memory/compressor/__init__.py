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
base.py                            — CompressorBase ABC
trainable_base.py                  — TrainableCompressorBase ABC
mcts_aware.py                      — MCTSAwareCompressor ABC (tree-aware opt-in)
llm_compressor.py                  — LLMCompressor (litellm + instructor; not trainable)
transformer_compressor.py          — TransformerCompressor (HF model; T5/BART/decoder)
hybrid_compressor.py               — HybridCompressor (slot extraction + free-form)
llm_mcts_compressor.py             — LLMMCTSCompressor (non-trainable MCTS baseline)
structured_selective_distiller.py  — StructuredSelectiveDistiller (Design 1: standalone)
mcts_gat_distiller.py              — MCTSGraphAttentionDistiller (Design 2: MCTS-aware)
tree_gat.py                        — PathSetEncoder (shared GAT for Design 2)
template.py                        — CompressedStateTemplate (section definitions + renderer)
lora_utils.py                      — inject_lora(), freeze_base_layers() helpers

Config types (configs/compressor/*.yaml)
-----------------------------------------
transformer          — plain T5 seq2seq baseline
llm                  — frozen LLM via litellm (not trainable)
hybrid               — slot extraction + narrative generation
llm_mcts             — LLM-based MCTS tree distillation (not trainable)
structured_selective — Design 1: section-aware cross-attention routing
mcts_gat             — Design 2: path-set attention over MCTS tree
identity             — pass-through (for ablation)
"""
