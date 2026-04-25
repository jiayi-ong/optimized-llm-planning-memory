"""Module tests for utils — episode_io save/load roundtrip workflows."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    EvalResult,
    RewardComponents,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.evaluation.manifest import EvalRunManifest
from optimized_llm_planning_memory.utils.episode_io import (
    list_episodes,
    list_eval_runs,
    load_eval_run,
    save_episode,
    save_eval_run,
)


def _make_episode(ep_id: str = "ep001", success: bool = True) -> EpisodeLog:
    traj = TrajectoryModel(
        trajectory_id=str(uuid.uuid4()),
        request_id="req-001",
        steps=(),
        total_steps=0,
    )
    reward = RewardComponents(
        hard_constraint_score=1.0,
        soft_constraint_score=0.8,
        tool_efficiency_score=0.9,
        tool_failure_penalty=0.0,
        logical_consistency_score=1.0,
        total_reward=0.85,
    )
    return EpisodeLog(
        episode_id=ep_id,
        request_id="req-001",
        agent_mode="raw",
        trajectory=traj,
        compressed_states=(),
        final_itinerary=None,
        reward_components=reward,
        tool_stats=(),
        total_steps=0,
        success=success,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _make_manifest(run_id: str) -> EvalRunManifest:
    return EvalRunManifest(
        run_id=run_id,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        compressor_type="identity",
        agent_mode="raw",
        judge_model_id="none",
        config_hash="manual",
        metric_version="v1",
        request_ids=["req-001"],
        n_episodes=1,
        deterministic_only=True,
    )


def _make_eval_result(ep_id: str = "ep001") -> EvalResult:
    return EvalResult(
        episode_id=ep_id,
        request_id="req-001",
        agent_mode="raw",
        deterministic_scores={"hard_constraint_ratio": 1.0, "overall_score": 0.9},
        llm_judge_scores={},
        overall_score=0.9,
        judge_model="none",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.mark.module_test
class TestEpisodeSaveLoadListWorkflow:
    def test_save_three_episodes_list_returns_three(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ep_id in ["ep001", "ep002", "ep003"]:
                save_episode(_make_episode(ep_id), tmpdir)
            episodes = list_episodes(tmpdir)
            assert len(episodes) == 3

    def test_episode_log_json_roundtrip_preserves_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ep = _make_episode("abc123", success=True)
            path = save_episode(ep, tmpdir)
            from optimized_llm_planning_memory.utils.episode_io import load_episode
            loaded = load_episode(path)
            assert loaded.episode_id == "abc123"
            assert loaded.success is True
            assert loaded.agent_mode == "raw"


@pytest.mark.module_test
class TestEvalRunSaveLoadWorkflow:
    def test_save_eval_run_creates_manifest_and_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest("run-abc")
            results = [_make_eval_result("ep001"), _make_eval_result("ep002")]
            run_dir = save_eval_run(manifest, results, tmpdir)
            assert (run_dir / "manifest.json").exists()
            assert (run_dir / "results.jsonl").exists()

    def test_load_eval_run_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest("run-xyz")
            results = [_make_eval_result()]
            save_eval_run(manifest, results, tmpdir)
            loaded_manifest, loaded_results = load_eval_run("run-xyz", tmpdir)
            assert loaded_manifest.run_id == "run-xyz"
            assert len(loaded_results) == 1
            assert loaded_results[0].overall_score == 0.9

    def test_multiple_eval_runs_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old = EvalRunManifest(
                run_id="old-run",
                created_at="2025-01-01T00:00:00+00:00",
                compressor_type="identity", agent_mode="raw",
                judge_model_id="none", config_hash="manual",
                metric_version="v1", request_ids=[], n_episodes=0,
                deterministic_only=True,
            )
            new = EvalRunManifest(
                run_id="new-run",
                created_at="2025-12-01T00:00:00+00:00",
                compressor_type="identity", agent_mode="raw",
                judge_model_id="none", config_hash="manual",
                metric_version="v1", request_ids=[], n_episodes=0,
                deterministic_only=True,
            )
            save_eval_run(old, [], tmpdir)
            save_eval_run(new, [], tmpdir)
            runs = list_eval_runs(tmpdir)
            assert runs[0].run_id == "new-run"
