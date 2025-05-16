#!/usr/bin/env python3
import os
import asyncio
import json
import sys
import argparse
from pathlib import Path
from loguru import logger
import instructor
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.formalization import FormalizationResponse, FormalizationPipeline

DEFAULT_LAKE_PATH = f"{os.path.expanduser('~')}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = f"{os.path.expanduser('~')}/code/mathlib4"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run autoformalization experiment with ablation studies"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable search component",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Disable feedback component",
    )
    return parser.parse_args()


def read_config(config_path: str) -> dict:
    """Read configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def process_problems(
    pipeline: FormalizationPipeline,
    input_path: Path,
    output_path: Path,
    enable_search: bool = True,
    enable_feedback: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
) -> None:
    """Process all problems from input file."""
    with open(input_path, "r") as f:
        ds = [json.loads(line) for line in f]

    if not ds:
        logger.error("No problems found in the input file. Exiting.")
        return

    if end_idx is None:
        end_idx = len(ds)
    ds = ds[start_idx:end_idx]

    logger.info(f"Processing {len(ds)} problems")
    successful = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(tqdm(ds)):
        logger.info(f"Processing problem {i+1}/{len(ds)}")

        try:
            resp = await pipeline.run(
                item["informal_problem"],
                enable_search=enable_search,
                enable_feedback=enable_feedback,
            )
            if resp:
                successful += 1

            output = {
                "id": item["id"],
                "informal_problem": item["informal_problem"],
                "formal_statement": resp.to_code() if resp else None,
            }
        except Exception as e:
            logger.error(f"Error processing problem {i}: {e}")
            output = {
                "id": item["id"],
                "informal_problem": item["informal_problem"],
                "formal_statement": None,
            }
        finally:
            with open(output_path, "a") as f:
                f.write(json.dumps(output) + "\n")

            if i < len(ds) - 1:
                await asyncio.sleep(1)

    logger.info(f"Batch processing complete. Success rate: {successful}/{len(ds)}")


async def main():
    """Main async function."""
    args = parse_args()
    config = read_config("configs/formalization.yaml")

    # Configure logging
    log_config = config.get("logging", {})
    log_path = log_config.get("out", "stdout")
    log_level = log_config.get("level", "INFO").upper()

    if log_path.lower() != "stdout":
        logger.remove()
        logger.add(log_path, level=log_level)

    logger.info("Starting Ablation Study")

    # Configure client
    config["client"]["mode"] = instructor.Mode[
        config["client"].get("mode", instructor.Mode.JSON)
    ]
    if config["client"]["engine"] == "anthropic":
        config["client"]["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif config["client"]["engine"] == "openai":
        config["client"]["api_key"] = os.environ.get("OPENAI_API_KEY")
    else:
        raise ValueError(f"Invalid engine: {config['client']['engine']}")

    # Initialize pipeline
    pipeline = FormalizationPipeline(
        client_config=config["client"],
        formalize_gen_config=config["formalize_gen"],
        search_gen_config=config["search_gen"],
        feedback_gen_config=config["feedback_gen"],
        search_config=config["search"],
        lake_path=config.get("lake_path", DEFAULT_LAKE_PATH),
        lean_workspace=config.get("lean_workspace", DEFAULT_LEAN_WORKSPACE),
        max_retries=config.get("max_retries", 0),
        num_max_iterations=config.get("num_max_iterations", 5),
    )

    # Both components enabled by default, unless explicitly disabled
    enable_search = not args.no_search
    enable_feedback = not args.no_feedback

    # Process problems
    await process_problems(
        pipeline=pipeline,
        input_path=Path(config["input_path"]),
        output_path=Path(config["output_path"]),
        enable_search=enable_search,
        enable_feedback=enable_feedback,
        start_idx=config.get("start_idx", 0),
        end_idx=config.get("end_idx"),
    )

    logger.info("Experiment complete")


if __name__ == "__main__":
    asyncio.run(main())
