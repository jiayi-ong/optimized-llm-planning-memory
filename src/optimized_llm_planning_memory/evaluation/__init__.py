"""
evaluation/
===========
Evaluation pipeline: deterministic scoring + LLM judge + ablation runner.

Design invariant
----------------
``DeterministicEvaluator`` imports ``ConstraintSatisfactionEngine`` from
``core/constraints.py`` — the **same class** used by ``training/reward.py``.
This guarantees that the training reward signal and the evaluation metric are
computed by identical logic. If this import ever diverges, the compressor will
optimise a proxy and fail at evaluation.

Contents
--------
evaluator.py     — Evaluator: orchestrates deterministic + LLM judge
deterministic.py — Hard constraint count, tool stats (uses ConstraintSatisfactionEngine)
llm_judge.py     — LLMJudge: rubric scoring via litellm + instructor
rubrics.py       — Rubric text constants
ablation.py      — AblationRunner: sweeps config axes, reruns Evaluator
"""
