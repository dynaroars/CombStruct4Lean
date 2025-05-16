import json
import os
import sys
import argparse
from tqdm.auto import tqdm
from loguru import logger
import concurrent.futures
import random
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.common import compile_lean4, DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE
from src.code_utils import remove_comments


logger.remove()
logger.add(sys.stdout, level="INFO")


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def read_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def _process_proofs_staged(
    original_proofs_list: list,
    item_success_dict: dict,
    executor: concurrent.futures.ThreadPoolExecutor,
    compile_fn,
):
    """
    Processes a list of proofs with random shuffling and staged submission.
    Updates pass@k metrics based on the shuffled order of successful attempts.
    """
    if not original_proofs_list:
        return

    shuffled_proofs = list(original_proofs_list)
    random.shuffle(shuffled_proofs)

    futures_map = {}

    if len(shuffled_proofs) > 0:
        idx = 0
        proof = shuffled_proofs[idx]
        future = executor.submit(compile_fn, proof)
        futures_map[future] = (proof, idx)

    for idx in range(1, min(5, len(shuffled_proofs))):
        proof = shuffled_proofs[idx]
        future = executor.submit(compile_fn, proof)
        futures_map[future] = (proof, idx)

    for idx in range(5, len(shuffled_proofs)):
        proof = shuffled_proofs[idx]
        future = executor.submit(compile_fn, proof)
        futures_map[future] = (proof, idx)

    for future in concurrent.futures.as_completed(futures_map):
        proof, shuffled_idx = futures_map[future]
        try:
            if future.result():
                item_success_dict["success"] = True
                item_success_dict["proofs"].append(proof)

                if shuffled_idx == 0:
                    item_success_dict["pass@1"] = True
                    item_success_dict["pass@5"] = True
                    item_success_dict["pass@10"] = True
                elif 1 <= shuffled_idx <= 4:
                    item_success_dict["pass@5"] = True
                    item_success_dict["pass@10"] = True
                elif shuffled_idx >= 5:
                    if shuffled_idx < 10:
                        item_success_dict["pass@10"] = True

        except Exception as exc:
            logger.error(
                f"Proof compilation generated an exception: {exc} for proof (shuffled index {shuffled_idx}): {proof[:100]}..."
            )


def process_item_validation(item):
    def compile_test(proof):
        if not proof:
            return False

        cleaned_proof = remove_comments(proof)
        contains_sorry = bool(re.search(r"\bsorry\b", cleaned_proof))

        if contains_sorry:
            logger.debug(
                f"Proof marked as failed due to 'sorry' keyword: {cleaned_proof[:100]}..."
            )
            return False

        if not cleaned_proof.strip():
            logger.debug("Proof is empty after comment removal.")
            return False

        try:
            status, _ = compile_lean4(
                cleaned_proof,
                lake_path=DEFAULT_LAKE_PATH,
                lean_workspace=DEFAULT_LEAN_WORKSPACE,
            )
            return status
        except Exception:
            return False

    item["success"] = {
        "success": False,
        "pass@1": False,
        "pass@5": False,
        "pass@10": False,
        "proofs": [],
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        _process_proofs_staged(
            original_proofs_list=item.get("proofs", []),
            item_success_dict=item["success"],
            executor=executor,
            compile_fn=compile_test,
        )
    return item


def main():
    parser = argparse.ArgumentParser(
        description="Validate Lean proofs from a JSONL file."
    )
    parser.add_argument("input_file", help="Path to input JSONL file containing proofs")
    args = parser.parse_args()

    logger.info(f"Reading data from: {args.input_file}")
    data = read_jsonl(args.input_file)
    data = data[:2]

    logger.info(f"Processing {len(data)} items...")
    processed_data = []
    for item in tqdm(data, desc="Processing items"):
        processed_data.append(process_item_validation(item))

    num_overall_success = 0
    num_pass_at_1 = 0
    num_pass_at_5 = 0
    num_pass_at_10 = 0

    for item in processed_data:
        if item["success"]["success"]:
            num_overall_success += 1
        if item["success"]["pass@1"]:
            num_pass_at_1 += 1
        if item["success"]["pass@5"]:
            num_pass_at_5 += 1
        if item["success"]["pass@10"]:
            num_pass_at_10 += 1

    total_items = len(processed_data)
    if total_items == 0:
        logger.info("No items processed.")
    else:
        logger.info(f"--- Statistics for {total_items} items ---")
        logger.info(
            f"Overall Success: {num_overall_success} ({num_overall_success/total_items:.2%})"
        )
        logger.info(f"Pass@1: {num_pass_at_1} ({num_pass_at_1/total_items:.2%})")
        logger.info(f"Pass@5: {num_pass_at_5} ({num_pass_at_5/total_items:.2%})")
        logger.info(f"Pass@10: {num_pass_at_10} ({num_pass_at_10/total_items:.2%})")
    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
