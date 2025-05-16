#!/usr/bin/env python3
import argparse
import os
import asyncio
import json
from pathlib import Path
import sys
from typing import List
from loguru import logger
import instructor
from tqdm import tqdm
import yaml
import random
from pydantic import BaseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.common import read_system_prompt, BaseGenerator
from src.code_utils import remove_comments

PROMPT = """**Original Problem:**
{}

**Grount truth formalization:**
{}

**Generated formalization:**
{}
"""


class NLIResponse(BaseModel):
    """Response for NLI task."""

    explanation: str
    label: bool


def read_config(config_path: str) -> dict:
    """Read configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run autoformalization experiment with ablation studies"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the file containing the generated formalizations",
    )
    parser.add_argument(
        "benchmark_filepath",
        type=str,
        help="Path to the file containing the benchmark data",
    )
    return parser.parse_args()


def load_and_process_data(filepath: str, benchmark_filepath: str):
    with open(benchmark_filepath, "r") as f:
        benchmark_data = [json.loads(line) for line in f.readlines()]
    benchmark_data = {item["id"]: item for item in benchmark_data}

    data = []
    with open(filepath, "r") as f:
        file_data = [json.loads(line) for line in f.readlines()]
    for item in file_data:
        if not item["compiled"]:
            continue
        data.append(
            {
                "id": item["id"],
                "formal_statement_pred": remove_comments(item["formal_statement_pred"]),
                "formal_statement": remove_comments(
                    benchmark_data[item["id"]]["formal_statement"]
                ),
                "informal_problem": benchmark_data[item["id"]]["informal_problem"],
                "filepath": filepath,
            }
        )
    return data


async def process_problems(
    generator: BaseGenerator,
    filepath: Path,
    benchmark_filepath: Path,
    output_path: Path,
    max_retries: int = 2,
) -> None:
    """Process all problems from input file."""
    data_to_eval = load_and_process_data(filepath, benchmark_filepath)

    logger.info(f"Processing {len(data_to_eval)} problems")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(tqdm(data_to_eval)):
        logger.info(f"Processing problem {i+1}/{len(data_to_eval)}")

        if not item.get("formal_statement") or not item.get("formal_statement_pred"):
            logger.warning(f"Skipping item {i} due to missing required fields")
            continue

        messages = [
            {"role": "system", "content": read_system_prompt("eval_alignment")},
            {
                "role": "user",
                "content": PROMPT.format(
                    item["informal_problem"],
                    item["formal_statement"],
                    item["formal_statement_pred"],
                ),
            },
        ]

        resp: NLIResponse = await generator.prompt(
            messages, response_model=NLIResponse, max_retries=max_retries
        )

        if resp:
            successful += 1
            output = {
                "id": item["id"],
                "formal_statement": item["formal_statement"],
                "formal_statement_pred": item["formal_statement_pred"],
                "informal_problem": item["informal_problem"],
                "compiled": item["compiled"],
                "explanation": resp.explanation,
                "aligned": resp.label,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(output) + "\n")

        if i < len(data_to_eval) - 1:
            await asyncio.sleep(1)

    logger.info(
        f"Batch processing complete. Success rate: {successful}/{len(data_to_eval)}"
    )


async def main():
    """Main async function."""
    args = parse_args()
    config = read_config("configs/semantic_check.yaml")

    log_config = config.get("logging", {})
    log_path = log_config.get("out", "stdout")
    log_level = log_config.get("level", "INFO").upper()

    if log_path.lower() != "stdout":
        logger.remove()
        logger.add(log_path, level=log_level)

    logger.info("Starting Semantic Check Process")

    config["client"]["mode"] = instructor.Mode[
        config["client"].get("mode", instructor.Mode.JSON)
    ]
    if config["client"]["engine"] == "anthropic":
        config["client"]["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
    elif config["client"]["engine"] == "openai":
        config["client"]["api_key"] = os.environ.get("OPENAI_API_KEY")
    else:
        raise ValueError(f"Invalid engine: {config['client']['engine']}")

    generator = BaseGenerator(client_config=config["client"], **config["generation"])

    await process_problems(
        generator=generator,
        filepath=Path(args.filepath),
        benchmark_filepath=Path(args.benchmark_filepath),
        output_path=Path(config["output_path"]),
        max_retries=config.get("max_retries", 2),
    )


if __name__ == "__main__":
    asyncio.run(main())
