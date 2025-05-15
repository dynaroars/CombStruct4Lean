#!/usr/bin/env python3
import os
import asyncio
import json
import sys
import yaml
from pathlib import Path
from loguru import logger
import instructor
from datasets import load_dataset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.formalization import FormalizationResponse, FormalizationPipeline

DEFAULT_LAKE_PATH = f"{os.path.expanduser('~')}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = f"{os.path.expanduser('~')}/code/mathlib4-deepseek"


def read_config(config_path: str) -> dict:
    """Read configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_example_problem() -> str:
    """Read the example problem from the markdown file."""
    prompt_path = Path("prompts/example_problem.md")
    with open(prompt_path, "r") as f:
        return f.read()


async def process_problems(
    pipeline: FormalizationPipeline,
    enable_search: bool,
    enable_feedback: bool,
    output_path: Path,
    start_idx: int = 0,
    end_idx: int = None,
    seed: int = 42,
) -> None:
    """Process all problems from the dataset."""
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    filtered_ds = ds.filter(
        lambda x: (
            x["problem_type"] == "Combinatorics"
            and x["problem_is_valid"] == "Yes"
            and x["solution_is_valid"] == "Yes"
            and x["question_type"] in ["math-word-problem", "proof", "other"]
            and x["answer"] == "proof"
        )
    )

    if end_idx is None:
        end_idx = len(filtered_ds)
    # Shuffle the dataset and sample it
    filtered_ds = filtered_ds.shuffle(seed=seed).select(range(start_idx, end_idx))

    logger.info(f"Processing {len(filtered_ds)} problems")
    successful = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(filtered_ds):
        logger.info(f"Processing problem {i+1}/{len(filtered_ds)}")

        resp = await pipeline.run(
            item["problem"],
            enable_search=enable_search,
            enable_feedback=enable_feedback,
        )

        if resp:
            successful += 1
            output = {
                "id": f"CombStruct4Lean_{i}",
                "informal_problem": item["problem"],
                "formal_statement": resp.to_code(),
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(output) + "\n")

        if i < len(filtered_ds) - 1:
            await asyncio.sleep(1)

    logger.info(
        f"Batch processing complete. Success rate: {successful}/{len(filtered_ds)}"
    )


async def main():
    """Main async function."""
    # Load configuration
    config = read_config("configs/formalization.yaml")

    # Configure logging
    log_config = config.get("logging", {})
    log_path = log_config.get("out", "stdout")
    log_level = log_config.get("level", "INFO").upper()

    if log_path.lower() != "stdout":
        logger.remove()
        logger.add(log_path, level=log_level)

    logger.info("Starting Formalizing Process")

    # Configure client
    config["client"]["mode"] = instructor.Mode[
        config["client"].get("mode", instructor.Mode.JSON)
    ]
    if config["client"]["engine"] == "anthropic":
        config["client"]["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif config["client"]["engine"] == "openai":
        config["client"]["api_key"] = os.environ.get("OPENAI_API_KEY")
    else:
        raise ValueError(f"Invalid engine: { config["client"]['engine']}")

    # Initialize pipeline
    pipeline = FormalizationPipeline(
        client_config=config["client"],
        formalize_gen_config=config["formalize_gen"],
        search_gen_config=config["search_gen"],
        feedback_gen_config=config["feedback_gen"],
        search_config=config["search_config"],
        lake_path=config.get("lake_path", DEFAULT_LAKE_PATH),
        lean_workspace=config.get("lean_workspace", DEFAULT_LEAN_WORKSPACE),
        max_retries=config.get("max_retries", 0),
        num_max_iterations=config.get("num_max_iterations", 5),
    )

    process_all = config.get("process_all", False)
    enable_search = config.get("enable_search", True)
    enable_feedback = config.get("enable_feedback", True)

    if process_all:
        await process_problems(
            pipeline=pipeline,
            enable_search=enable_search,
            enable_feedback=enable_feedback,
            output_path=Path(config["output_path"]),
            start_idx=config.get("start_idx", 0),
            end_idx=config.get("end_idx", None),
            seed=config.get("seed", 42),
        )
    else:
        # Process example problem
        logger.info("Using example problem")
        problem_text = read_example_problem()
        resp = await pipeline.run(problem_text)
        logger.info(f"Formalized problem: {resp.to_code()}")
        logger.info("Formalization complete")


if __name__ == "__main__":
    asyncio.run(main())
