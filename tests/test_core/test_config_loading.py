"""
tests/test_core/test_config_loading.py
=======================================
T5 — YAML config integration tests.

Load every YAML file under configs/ and validate it against the matching
Pydantic config model.  Ensures that hand-edited YAML stays in sync with
the Python schema — a config typo would be caught here rather than at
runtime during a training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from optimized_llm_planning_memory.core.config import (
    AgentConfig,
    EvalConfig,
    RewardConfig,
    TrainingConfig,
)

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def _load_yaml(path: Path) -> dict:
    """Return the inner content dict (strip the top-level @package key)."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Hydra yaml files have a single top-level key (e.g. "agent:", "reward:")
    if raw and len(raw) == 1:
        return list(raw.values())[0] or {}
    return raw or {}


@pytest.mark.unit
class TestAgentConfigYAML:
    @pytest.mark.parametrize("yaml_file", list((CONFIGS_DIR / "agent").glob("*.yaml")))
    def test_agent_yaml_validates(self, yaml_file):
        data = _load_yaml(yaml_file)
        config = AgentConfig.model_validate(data)
        assert config.max_steps >= 1
        assert config.llm_model_id  # non-empty string

    def test_default_system_prompt_version_is_v2(self):
        """L4 fix: system_prompt_version default must be v2, not v1."""
        data = _load_yaml(CONFIGS_DIR / "agent" / "react_default.yaml")
        config = AgentConfig.model_validate(data)
        assert config.system_prompt_version == "v2", (
            "react_default.yaml must use v2 prompt (L4 fix). "
            f"Got: {config.system_prompt_version}"
        )


@pytest.mark.unit
class TestRewardConfigYAML:
    def test_reward_default_yaml_validates(self):
        data = _load_yaml(CONFIGS_DIR / "reward" / "default.yaml")
        config = RewardConfig.model_validate(data)
        assert config.weights is not None


@pytest.mark.unit
class TestTrainingConfigYAML:
    @pytest.mark.parametrize("yaml_file", list((CONFIGS_DIR / "training").glob("*.yaml")))
    def test_training_yaml_validates(self, yaml_file):
        data = _load_yaml(yaml_file)
        config = TrainingConfig.model_validate(data)
        assert config.num_timesteps >= 1


@pytest.mark.unit
class TestEvalConfigYAML:
    def test_eval_default_yaml_validates(self):
        data = _load_yaml(CONFIGS_DIR / "eval" / "default.yaml")
        # eval config has extra keys (scoring_weights) — use strict=False
        filtered = {k: v for k, v in data.items() if k in EvalConfig.model_fields}
        config = EvalConfig.model_validate(filtered)
        assert config.rubric_path
