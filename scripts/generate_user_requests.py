"""
scripts/generate_user_requests.py
===================================
Generate synthetic UserRequest JSON files for train/val/test splits.

Uses an LLM (via litellm + instructor) to produce diverse travel scenarios
with hard and soft constraints, then validates each against the UserRequest
Pydantic schema before writing to disk.

Usage
-----
    python scripts/generate_user_requests.py
    python scripts/generate_user_requests.py n_train=200 n_val=50 n_test=50
    python scripts/generate_user_requests.py model=openai/gpt-4o-mini seed=123
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf

from optimized_llm_planning_memory.utils.logging import configure_logging, get_logger
from optimized_llm_planning_memory.utils.seed import set_seed


GENERATION_PROMPT = """\
Generate a realistic travel planning request in JSON format. The request should
include a natural language description, origin and destination cities, date range,
budget, traveler profile, and 2–4 hard constraints plus 2–3 soft constraints.

Hard constraint categories: budget, date, duration, city, accommodation, transport.
Soft constraint categories: accommodation, activity, preference, accessibility.

Return valid JSON matching this schema:
- request_id: string (UUID)
- raw_text: string (natural language request)
- origin_city: string
- destination_cities: list of strings (1–3 cities)
- start_date: YYYY-MM-DD
- end_date: YYYY-MM-DD
- budget_usd: float
- traveler_profile: {num_adults, num_children, accessibility_needs, dietary_restrictions}
- hard_constraints: list of {constraint_id, constraint_type: "hard", category, description, value, unit}
- soft_constraints: list of {constraint_id, constraint_type: "soft", category, description, value, unit}
- preferences: list of strings

Make the scenario specific and varied — different destinations, budgets, group sizes,
and constraint combinations each time. Date ranges should be in 2025.
"""


def generate_requests(
    n: int,
    model_id: str,
    split: str,
    output_dir: Path,
    log,
) -> None:
    """Generate ``n`` UserRequest JSON files into ``output_dir``."""
    import litellm
    import instructor
    from optimized_llm_planning_memory.core.models import UserRequest

    client = instructor.from_litellm(litellm.completion)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        try:
            result = client.chat.completions.create(
                model=model_id,
                response_model=UserRequest,
                messages=[
                    {"role": "system", "content": "You are a travel scenario generator."},
                    {"role": "user", "content": GENERATION_PROMPT},
                ],
                temperature=1.0,
            )
            # Assign a fresh UUID to avoid collisions
            result = result.model_copy(update={"request_id": str(uuid.uuid4())})
            out_path = output_dir / f"request_{result.request_id}.json"
            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            log.info("generated", split=split, index=i + 1, total=n, request_id=result.request_id)
        except Exception as exc:
            log.warning("generation_failed", split=split, index=i + 1, error=str(exc))


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.logging.level)
    log = get_logger(__name__)
    set_seed(cfg.project.seed)

    # Allow CLI overrides: n_train=200 n_val=50 n_test=50 model=...
    n_train = int(OmegaConf.select(cfg, "n_train", default=100))
    n_val = int(OmegaConf.select(cfg, "n_val", default=25))
    n_test = int(OmegaConf.select(cfg, "n_test", default=25))
    model = OmegaConf.select(cfg, "model", default=cfg.agent.llm_model_id)

    base = Path("data/user_requests")
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        log.info("generating_split", split=split, n=n, model=model)
        generate_requests(n, model, split, base / split, log)

    log.info("done", total=n_train + n_val + n_test)


if __name__ == "__main__":
    main()
