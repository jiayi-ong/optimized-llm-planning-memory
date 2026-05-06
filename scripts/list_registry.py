"""
scripts/list_registry.py
========================
Print the prompt and augmentation registries to stdout.

Usage
-----
    # Show all entries
    python scripts/list_registry.py

    # Filter to specific IDs (useful from notebook config preview cell)
    python scripts/list_registry.py --prompt-id v2 --aug-id ssd-init-001

    # Show prompts only
    python scripts/list_registry.py --prompts-only

    # Show augmentations only
    python scripts/list_registry.py --augs-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _fmt_bool(v: bool) -> str:
    return "yes" if v else "no"


def _col(text: str, width: int) -> str:
    return str(text)[:width].ljust(width)


def print_prompts(prompt_registry, filter_id: str | None = None) -> None:
    entries = prompt_registry.list()
    if filter_id:
        entries = [e for e in entries if e.prompt_id == filter_id]

    header = (
        f"{'PROMPT_ID':<18}  {'TEXT_REF':<12}  {'DEPRECATED':<10}  DESCRIPTION"
    )
    print("\n-- Prompts -----------------------------------------------------------------")
    print(header)
    print("-" * 80)
    for e in sorted(entries, key=lambda x: x.prompt_id):
        dep = "[deprecated]" if e.deprecated else ""
        print(
            f"{_col(e.prompt_id, 18)}  {_col(e.text_ref, 12)}  "
            f"{_col(dep, 10)}  {e.description[:60]}"
        )
        if e.notes:
            print(f"  {'':18}   Notes: {e.notes[:72]}")

    if not entries:
        print(f"  (no entries match filter '{filter_id}')")


def print_augmentations(aug_registry, filter_id: str | None = None) -> None:
    entries = aug_registry.list()
    if filter_id:
        entries = [e for e in entries if e.aug_id == filter_id]

    print("\n-- Augmentations -----------------------------------------------------------")
    header = (
        f"{'AUG_ID':<26}  {'TYPE':<22}  {'CKPT?':<6}  "
        f"{'AGENT_LLM':<28}  DESCRIPTION"
    )
    print(header)
    print("-" * 110)
    for e in sorted(entries, key=lambda x: x.aug_id):
        dep = " [dep]" if e.deprecated else ""
        ckpt = "yes" if e.has_checkpoint else "no"
        agent_llm = e.config_snapshot.get("agent_llm_model_id", "-")
        print(
            f"{_col(e.aug_id + dep, 26)}  {_col(e.type.value, 22)}  "
            f"{_col(ckpt, 6)}  {_col(agent_llm, 28)}  {e.description[:46]}"
        )
        if e.checkpoint_path:
            print(f"  {'':26}   ckpt: {e.checkpoint_path}")
        if e.config_snapshot.get("model_name_or_path"):
            print(f"  {'':26}   base: {e.config_snapshot['model_name_or_path']}")
        if e.notes:
            print(f"  {'':26}   note: {e.notes[:72]}")

    if not entries:
        print(f"  (no entries match filter '{filter_id}')")


def main() -> None:
    parser = argparse.ArgumentParser(description="List prompt and augmentation registries.")
    parser.add_argument("--prompt-id", default=None, help="Show only this prompt ID")
    parser.add_argument("--aug-id", default=None, help="Show only this augmentation ID")
    parser.add_argument("--prompts-only", action="store_true")
    parser.add_argument("--augs-only", action="store_true")
    args = parser.parse_args()

    from optimized_llm_planning_memory.core.registry import AugmentationRegistry, PromptRegistry

    prompt_registry = PromptRegistry.load()
    aug_registry = AugmentationRegistry.load()

    if not args.augs_only:
        print_prompts(prompt_registry, filter_id=args.prompt_id)

    if not args.prompts_only:
        print_augmentations(aug_registry, filter_id=args.aug_id)

    print()


if __name__ == "__main__":
    main()
