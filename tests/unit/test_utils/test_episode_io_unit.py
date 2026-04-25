"""Unit tests for utils/episode_io.py — episode and eval run persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from optimized_llm_planning_memory.core.models import (
    EpisodeLog,
    EvalResult,
    RewardComponents,
    TrajectoryModel,
)
from optimized_llm_planning_memory.mcts.node import MCTSStats  # resolve EpisodeLog forward ref
EpisodeLog.model_rebuild()
from optimized_llm_planning_memory.evaluation.manifest import EvalRunManifest
from optimized_llm_planning_memory.utils.episode_io import (
    list_episodes,
    list_eval_runs,
    load_episode,
    load_eval_run,
    save_episode,
    save_eval_run,
)


def _make_episode(ep_id: str = "ep001") -> EpisodeLog:
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
        success=True,
        config_hash="test",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


def _make_manifest(run_id: str = "run001") -> EvalRunManifest:
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


def _make_eval_result() -> EvalResult:
    return EvalResult(
        episode_id="ep001",
        request_id="req-001",
        agent_mode="raw",
        deterministic_scores={"hard_constraint_ratio": 1.0},
        llm_judge_scores={},
        overall_score=0.9,
        judge_model="none",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@pytest.mark.unit
class TestSaveAndLoadEpisode:
    def test_save_creates_file(self):
        ep = _make_episode("abc")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_episode(ep, tmpdir)
            assert path.exists()
            assert path.name == "ep_abc.json"

    def test_load_roundtrip(self):
        ep = _make_episode("xyz")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_episode(ep, tmpdir)
            loaded = load_episode(path)
            assert loaded.episode_id == "xyz"
            assert loaded.success == ep.success

    def test_save_creates_directory_if_missing(self):
        ep = _make_episode("newdir")
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "episodes" / "nested"
            save_episode(ep, subdir)
            assert (subdir / "ep_newdir.json").exists()


@pytest.mark.unit
class TestListEpisodes:
    def test_empty_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert list_episodes(tmpdir) == []

    def test_nonexistent_directory_returns_empty_list(self):
        assert list_episodes("/nonexistent/path/abc") == []

    def test_list_returns_all_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_episode(_make_episode("a1"), tmpdir)
            save_episode(_make_episode("a2"), tmpdir)
            save_episode(_make_episode("a3"), tmpdir)
            episodes = list_episodes(tmpdir)
            assert len(episodes) == 3

    def test_corrupted_file_skipped_silently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_episode(_make_episode("good"), tmpdir)
            bad = Path(tmpdir) / "ep_bad.json"
            bad.write_text("not valid json", encoding="utf-8")
            episodes = list_episodes(tmpdir)
            assert len(episodes) == 1


@pytest.mark.unit
class TestSaveAndLoadEvalRun:
    def test_save_creates_manifest_and_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest("run001")
            results = [_make_eval_result()]
            run_dir = save_eval_run(manifest, results, tmpdir)
            assert (run_dir / "manifest.json").exists()
            assert (run_dir / "results.jsonl").exists()

    def test_load_eval_run_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _make_manifest("run002")
            results = [_make_eval_result()]
            save_eval_run(manifest, results, tmpdir)
            loaded_manifest, loaded_results = load_eval_run("run002", tmpdir)
            assert loaded_manifest.run_id == "run002"
            assert len(loaded_results) == 1

    def test_load_missing_run_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_eval_run("nonexistent_run", tmpdir)


@pytest.mark.unit
class TestListEvalRuns:
    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert list_eval_runs(tmpdir) == []

    def test_nonexistent_directory_returns_empty(self):
        assert list_eval_runs("/nonexistent/eval_runs") == []

    def test_multiple_runs_returned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for run_id in ["r1", "r2", "r3"]:
                save_eval_run(_make_manifest(run_id), [], tmpdir)
            runs = list_eval_runs(tmpdir)
            assert len(runs) == 3

    def test_runs_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m1 = EvalRunManifest(
                run_id="old",
                created_at="2025-01-01T00:00:00+00:00",
                compressor_type="identity", agent_mode="raw",
                judge_model_id="none", config_hash="manual",
                metric_version="v1", request_ids=[], n_episodes=0,
                deterministic_only=True,
            )
            m2 = EvalRunManifest(
                run_id="new",
                created_at="2025-06-01T00:00:00+00:00",
                compressor_type="identity", agent_mode="raw",
                judge_model_id="none", config_hash="manual",
                metric_version="v1", request_ids=[], n_episodes=0,
                deterministic_only=True,
            )
            save_eval_run(m1, [], tmpdir)
            save_eval_run(m2, [], tmpdir)
            runs = list_eval_runs(tmpdir)
            assert runs[0].run_id == "new"
