#!/usr/bin/env python3
import os
import asyncio
import json
from pathlib import Path
import sys
from loguru import logger
import instructor
from tqdm import tqdm
import yaml
import random
from pydantic import BaseModel

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.common import read_system_prompt, BaseGenerator

PROMPT = """**Problem A:**
{}

**Problem B:**
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


async def process_problems(
    generator: BaseGenerator,
    input_path: Path,
    output_path: Path,
    start_idx: int = 0,
    end_idx: int = None,
    max_retries: int = 2,
) -> None:
    """Process all problems from input file."""
    with open(input_path, "r") as f:
        ds = [json.loads(line.strip()) for line in f]

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

        if not item.get("formal_statement") or not item.get("backtranslated_informal"):
            logger.warning(f"Skipping item {i} due to missing required fields")
            continue

        # Randomly shuffle the order of comparison
        elements = [item["informal_problem"], item["backtranslated_informal"]]
        random.shuffle(elements)

        messages = [
            {"role": "system", "content": read_system_prompt("semantic_check")},
            {"role": "user", "content": PROMPT.format(elements[0], elements[1])},
        ]

        resp: NLIResponse = await generator.prompt(
            messages, response_model=NLIResponse, max_retries=max_retries
        )

        if resp:
            successful += 1
            output = {
                "id": item["id"],
                "formal_statement": item["formal_statement"],
                "backtranslated_informal": item["backtranslated_informal"],
                "informal_problem": item["informal_problem"],
                "explanation": resp.explanation,
                "label": resp.label,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(output) + "\n")

        if i < len(ds) - 1:
            await asyncio.sleep(1)

    logger.info(f"Batch processing complete. Success rate: {successful}/{len(ds)}")


async def main():
    """Main async function."""
    # Load configuration
    config = read_config("configs/semantic_check.yaml")

    # Configure logging
    log_config = config.get("logging", {})
    log_path = log_config.get("out", "stdout")
    log_level = log_config.get("level", "INFO").upper()

    if log_path.lower() != "stdout":
        logger.remove()
        logger.add(log_path, level=log_level)

    logger.info("Starting Semantic Check Process")

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

    # Initialize generator
    generator = BaseGenerator(client_config=config["client"], **config["generation"])

    await process_problems(
        generator=generator,
        input_path=Path(config["input_path"]),
        output_path=Path(config["output_path"]),
        start_idx=config.get("start_idx", 0),
        end_idx=config.get("end_idx"),
        max_retries=config.get("max_retries", 2),
    )


if __name__ == "__main__":
    asyncio.run(main())
