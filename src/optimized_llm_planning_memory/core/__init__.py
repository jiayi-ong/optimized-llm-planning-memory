"""
core/
=====
The zero-dependency foundation of the project.

Design note
-----------
This package imports ONLY from Python's standard library and third-party packages
(pydantic, pydantic-settings). No other internal sub-packages import from here;
all other sub-packages import FROM here. This strict DAG prevents circular imports
and makes ``core/`` trivially unit-testable in isolation.

Contents
--------
models.py       — All Pydantic v2 data models shared across modules.
constraints.py  — ConstraintSatisfactionEngine: the SINGLE implementation of
                  constraint scoring used by both reward and evaluation.
config.py       — Project-wide configuration schema (pydantic-settings + Hydra compat).
exceptions.py   — Custom exception hierarchy.
"""
