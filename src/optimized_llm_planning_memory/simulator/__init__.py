"""
simulator/
==========
Boundary layer between this project and the external travel simulator library.

Design principle: Stable abstraction over an unstable dependency
----------------------------------------------------------------
The external library may change its API, return types, or class names between
versions. By wrapping it behind ``SimulatorProtocol`` and ``SimulatorAdapter``,
the rest of the codebase is completely insulated from those changes. Only this
package ever imports from the external library.

Responsibilities
----------------
SimulatorProtocol   — ``typing.Protocol`` defining the interface contract.
                      Other modules type-hint against this; never against the
                      concrete adapter or the external library.

SimulatorAdapter    — Wraps the external library. Translates its return types
                      into our Pydantic schemas (``schemas.py``). Contains NO
                      business logic, tracking, or error feedback generation —
                      those concerns belong in ``tools/``.

schemas.py          — Pydantic models for simulator request/response shapes.
                      These act as a versioned contract between the adapter and
                      the rest of the system.

What is NOT here
----------------
* HTTP client or connection pool — the simulator is an in-process Python library.
* Tracking or usage metrics — see ``tools/tracker.py``.
* Retry logic — see ``tools/base.py``.
* Error feedback for the agent — see ``tools/base.py``.
"""
