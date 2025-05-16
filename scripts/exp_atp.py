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

from src.common import BaseGenerator, read_system_prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run automated theorem proving experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/theorem_proving.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--num-responses",
        type=int,
        default=1,
        help="Number of proof attempts to generate per problem",
    )
    return parser.parse_args()


def read_config(config_path: str) -> dict:
    """Read configuration from yaml file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def llm_validator(
    template: str,
    client: instructor.Instructor,
    allow_override: bool = False,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: int = 1024,
    **kwargs,
) -> callable:
    """Create a validator function for LLM responses."""
    rule_prompt = """You are given two pieces of formal code: A (a template) and B (a proposed solution).
Verify if B is a valid solution for A by checking the following:
1.  **Presence**: All definitions and theorems specified in template A must also be present in solution B.
2.  **Signature Integrity**: Solution B must not alter the signatures (including types, parameters, and formal statements) of any definitions or theorems present in template A.
3.  **No Overwriting**: While solution B may introduce new definitions and lemmas, it must not redefine or overwrite any existing definitions or theorems from template A.
4.  **Completeness**: All proofs in solution B corresponding to theorems or tasks in template A must be complete, with no placeholders (e.g., `sorry`, `todo`, `admit`).

Evaluate if B satisfies all these conditions with respect to A. Ignore any extra definitions or lemmas in B that do not correspond to template A, as long as they don't violate the above rules."""

    def llm(v: str) -> str:
        logger.debug(f"Validating response: {v}")
        resp = client.chat.completions.create(
            response_model=instructor.dsl.validators.Validator,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world class validation model. Capable to determine if the following value is valid for the statement, if it is not, explain why and suggest a new value.",
                },
                {
                    "role": "user",
                    "content": f"{rule_prompt}\n\nTEMPLATE:\n{template}\n\nPROPOSED SOLUTION:\n{v}",
                },
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(f"Validation result: {resp}")

        assert resp.is_valid, resp.reason

        if allow_override and not resp.is_valid and resp.fixed_value is not None:
            return resp.fixed_value
        return v

    return llm


async def generate_and_validate_response(
    generator: BaseGenerator,
    messages: list[dict[str, str]],
    validator: callable,
    max_retries: int = 0,
) -> str | None:
    """Generates a response and validates it."""
    try:
        resp: str = await generator.prompt(
            messages, response_model=str, max_retries=max_retries
        )
        return validator(resp)
    except Exception as e:
        logger.warning(f"Generation or validation failed: {e}")
        return None


async def process_problems(
    generator: BaseGenerator,
    input_path: Path,
    output_path: Path,
    validator_config: dict,
    num_responses: int = 1,
    max_retries: int = 0,
    start_idx: int = 0,
    end_idx: int = None,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(tqdm(ds)):
        logger.info(f"Processing problem {i+1}/{len(ds)}")

        if not item.get("formal_statement"):
            logger.warning(f"Skipping item {i} due to missing formal statement")
            continue

        template = item["formal_statement"]
        validator_func = llm_validator(template, generator.client, **validator_config)

        messages = [
            {"role": "system", "content": read_system_prompt("theorem_proving")},
            {"role": "user", "content": template},
        ]

        tasks = [
            generate_and_validate_response(
                generator, messages, validator_func, max_retries
            )
            for _ in range(num_responses)
        ]

        proofs = [proof for proof in await asyncio.gather(*tasks) if proof is not None]

        output = {
            "id": item["id"],
            "formal_statement": template,
            "proofs": proofs,
            "informal_problem": item["informal_problem"],
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(output) + "\n")

        if i < len(ds) - 1:
            await asyncio.sleep(1)

    logger.info("Processing complete")


async def main():
    """Main async function."""
    args = parse_args()
    config = read_config(args.config)

    log_config = config.get("logging", {})
    log_path = log_config.get("out", "stdout")
    log_level = log_config.get("level", "INFO").upper()

    if log_path.lower() != "stdout":
        logger.remove()
        logger.add(log_path, level=log_level)

    logger.info("Starting Theorem Proving Process")

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
        input_path=Path(config["input_path"]),
        output_path=Path(config["output_path"]),
        validator_config=config["validator"],
        num_responses=args.num_responses,
        max_retries=config.get("max_retries", 0),
        start_idx=config.get("start_idx", 0),
        end_idx=config.get("end_idx"),
    )


if __name__ == "__main__":
    asyncio.run(main())
