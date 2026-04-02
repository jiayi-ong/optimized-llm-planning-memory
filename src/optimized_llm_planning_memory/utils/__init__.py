"""
utils/
======
Cross-cutting utilities: logging, TensorBoard helpers, visualization,
reproducible seeding, and episode I/O.

Contents
--------
logging.py        — structlog setup with JSON + console sinks
tensorboard.py    — Typed TensorBoard writer helpers
visualization.py  — Pretty-print ReAct steps, CompressedState, rewards
seed.py           — set_seed(): Python + numpy + torch reproducibility
episode_io.py     — EpisodeLog ↔ JSON file (save / load / list)
"""
