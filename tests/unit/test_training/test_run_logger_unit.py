"""
Unit tests — RLRunLogger JSONL round-trip and TrainingRunManifest.

Verifies:
1. RLRunLogger creates files and writes valid JSONL.
2. load_ppo_metrics / load_episode_metrics correctly deserialise records.
3. TrainingRunManifest.create() saves and loads correctly.
4. resolve_checkpoint() finds the final checkpoint zip.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from optimized_llm_planning_memory.training.run_logger import (
    EpisodeMetricsSummary,
    PPOUpdateMetrics,
    RLRunLogger,
    list_run_ids,
    load_episode_metrics,
    load_ppo_metrics,
)
from optimized_llm_planning_memory.training.run_manifest import (
    TrainingRunManifest,
    load_manifest,
    resolve_checkpoint,
    save_manifest,
)


@pytest.mark.unit_test
class TestRLRunLoggerJSONL:
    def test_logger_creates_jsonl_files(self, tmp_path):
        logger = RLRunLogger(run_id="test_run", training_dir=tmp_path)
        logger.close()
        assert (tmp_path / "test_run" / "ppo_metrics.jsonl").exists()
        assert (tmp_path / "test_run" / "episode_metrics.jsonl").exists()

    def test_log_ppo_update_writes_line(self, tmp_path):
        logger = RLRunLogger(run_id="test_run", training_dir=tmp_path)
        metrics = PPOUpdateMetrics(
            update_step=0,
            policy_loss=-0.05,
            value_loss=0.1,
            entropy_loss=-0.02,
            total_loss=0.08,
            clip_fraction=0.12,
            approx_kl=0.01,
            explained_variance=0.4,
            learning_rate=3e-4,
        )
        logger.log_ppo_update(metrics)
        logger.close()

        records = load_ppo_metrics("test_run", tmp_path)
        assert len(records) == 1
        assert records[0].update_step == 0
        assert abs(records[0].policy_loss - (-0.05)) < 1e-6

    def test_log_episode_summary_writes_line(self, tmp_path):
        logger = RLRunLogger(run_id="test_run", training_dir=tmp_path)
        summary = EpisodeMetricsSummary(
            episode_id="ep-001",
            request_id="req-001",
            agent_mode="compressor",
            total_steps=10,
            success=True,
            total_reward=0.75,
            hard_constraint_score=0.9,
            soft_constraint_score=0.8,
            tool_efficiency_score=0.85,
            tool_failure_penalty=0.0,
            logical_consistency_score=1.0,
        )
        logger.log_episode_summary(summary)
        logger.close()

        records = load_episode_metrics("test_run", tmp_path)
        assert len(records) == 1
        assert records[0].episode_id == "ep-001"
        assert records[0].success is True

    def test_multiple_records_round_trip(self, tmp_path):
        logger = RLRunLogger(run_id="multi_run", training_dir=tmp_path)
        for i in range(5):
            logger.log_ppo_update(PPOUpdateMetrics(
                update_step=i,
                policy_loss=-0.01 * i,
                value_loss=0.1,
                entropy_loss=-0.01,
                total_loss=0.09,
                clip_fraction=0.15,
                approx_kl=0.01,
                explained_variance=0.5,
                learning_rate=3e-4,
            ))
        logger.close()

        records = load_ppo_metrics("multi_run", tmp_path)
        assert len(records) == 5
        assert [r.update_step for r in records] == list(range(5))

    def test_list_run_ids_returns_newest_first(self, tmp_path):
        for run_id in ["20260101_000000", "20260102_000000", "20260103_000000"]:
            logger = RLRunLogger(run_id=run_id, training_dir=tmp_path)
            logger.log_ppo_update(PPOUpdateMetrics(
                update_step=0, policy_loss=0.0, value_loss=0.0, entropy_loss=0.0,
                total_loss=0.0, clip_fraction=0.0, approx_kl=0.0,
                explained_variance=0.0, learning_rate=0.0,
            ))
            logger.close()

        ids = list_run_ids(tmp_path)
        assert ids[0] == "20260103_000000"

    def test_context_manager_closes_gracefully(self, tmp_path):
        with RLRunLogger(run_id="ctx_run", training_dir=tmp_path) as logger:
            logger.log_ppo_update(PPOUpdateMetrics(
                update_step=0, policy_loss=0.0, value_loss=0.0, entropy_loss=0.0,
                total_loss=0.0, clip_fraction=0.0, approx_kl=0.0,
                explained_variance=0.0, learning_rate=0.0,
            ))
        # File should be flushed and closed
        records = load_ppo_metrics("ctx_run", tmp_path)
        assert len(records) == 1

    def test_grad_norm_optional_field(self, tmp_path):
        logger = RLRunLogger(run_id="grad_run", training_dir=tmp_path)
        logger.log_ppo_update(PPOUpdateMetrics(
            update_step=0, policy_loss=0.0, value_loss=0.0, entropy_loss=0.0,
            total_loss=0.0, clip_fraction=0.0, approx_kl=0.0,
            explained_variance=0.0, learning_rate=0.0,
            grad_norm=0.85,
            advantages_mean=0.1,
            advantages_std=0.3,
        ))
        logger.close()
        records = load_ppo_metrics("grad_run", tmp_path)
        assert records[0].grad_norm == pytest.approx(0.85)
        assert records[0].advantages_mean == pytest.approx(0.1)
        assert records[0].advantages_std == pytest.approx(0.3)


@pytest.mark.unit_test
class TestTrainingRunManifest:
    def test_create_and_save_and_load(self, tmp_path):
        manifest = TrainingRunManifest.create(
            run_id="20260501_120000",
            compressor_type="IdentityCompressor",
            n_train_requests=40,
            checkpoint_dir=tmp_path / "checkpoints",
            run_name="test_run",
            agent_mode="compressor",
            reward_weights={"hard_constraint": 2.0, "soft_constraint": 1.0},
            ppo_hyperparams={"learning_rate": 3e-5, "n_steps": 256},
            n_envs=2,
            num_timesteps=50000,
        )
        save_manifest(manifest, tmp_path / "training" / "20260501_120000")
        loaded = load_manifest("20260501_120000", tmp_path / "training")

        assert loaded is not None
        assert loaded.run_id == "20260501_120000"
        assert loaded.compressor_type == "IdentityCompressor"
        assert loaded.n_train_requests == 40
        assert loaded.reward_weights["hard_constraint"] == pytest.approx(2.0)

    def test_load_missing_manifest_returns_none(self, tmp_path):
        result = load_manifest("nonexistent_run", tmp_path)
        assert result is None

    def test_resolve_checkpoint_finds_final_zip(self, tmp_path):
        final_dir = tmp_path / "checkpoints" / "final"
        final_dir.mkdir(parents=True)
        zip_path = final_dir / "ppo_model.zip"
        zip_path.write_text("fake zip")

        resolved = resolve_checkpoint("any_run_id", output_dir=tmp_path)
        assert resolved == zip_path

    def test_resolve_checkpoint_finds_numbered_zip_when_no_final(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        zip1 = ckpt_dir / "ppo_compressor_5000_steps.zip"
        zip2 = ckpt_dir / "ppo_compressor_10000_steps.zip"
        zip1.write_text("fake")
        zip2.write_text("fake")

        resolved = resolve_checkpoint("any_run", output_dir=tmp_path, prefer_final=False)
        assert resolved == zip2  # latest by step count

    def test_resolve_checkpoint_returns_none_when_nothing_exists(self, tmp_path):
        result = resolve_checkpoint("no_run", output_dir=tmp_path)
        assert result is None
