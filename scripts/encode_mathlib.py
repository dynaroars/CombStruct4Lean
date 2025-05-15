import argparse
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any
import sys
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
from loguru import logger
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.code_utils import extract_core_definition, remove_comments


def process_lean_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads a Lean file, removes comments, splits into definitions,
    extracts core definitions, and returns a list of results.
    """
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

    cleaned_content = remove_comments(content)
    blocks = cleaned_content.split("\n\n")

    for block in blocks:
        trimmed_block = block.strip()
        if not trimmed_block:
            continue

        core_def = extract_core_definition(trimmed_block)
        if core_def:
            results.append(
                {
                    "filepath": str(file_path),
                    "code": trimmed_block,
                    "extracted_core_definition": core_def,
                }
            )
    return results


def extract_definitions(input_dir: Path) -> List[Dict[str, Any]]:
    """Extract definitions from all Lean files in the input directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

    all_results = []
    lean_files = list(input_dir.rglob("*.lean"))

    logger.info(f"Found {len(lean_files)} .lean files in {input_dir}")

    for lean_file in tqdm(lean_files, desc="Processing Lean files"):
        file_results = process_lean_file(lean_file)
        all_results.extend(file_results)

    logger.info(f"Extracted {len(all_results)} definitions")
    return all_results


def encode_definitions(
    data: List[Dict[str, Any]], model_name: str
) -> List[Dict[str, Any]]:
    """Encode the extracted definitions using a Sentence Transformer model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    processed_data = []
    skipped_count = 0

    for example in tqdm(data, desc="Encoding definitions"):
        core_def = example.get("extracted_core_definition")
        code_text = example.get("code")

        if not isinstance(core_def, str) or not isinstance(code_text, str):
            logger.debug(f"Skipping invalid example")
            skipped_count += 1
            continue

        processed_example = example.copy()
        processed_example["core_definition_embedding"] = model.encode(
            core_def, convert_to_numpy=True
        )
        processed_example["code_embedding"] = model.encode(
            code_text, convert_to_numpy=True
        )
        processed_data.append(processed_example)

    logger.info(f"Finished encoding. Skipped {skipped_count} examples")
    return processed_data


def main():
    parser = argparse.ArgumentParser(
        description="Extract and encode definitions from Lean files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .lean files")
    parser.add_argument(
        "output_pickle", type=str, help="Path to save the output pickle file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nomic-ai/CodeRankEmbed",
        help="Name of the Sentence Transformer model",
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_pickle)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        definitions = extract_definitions(input_path)
        encoded_data = encode_definitions(definitions, args.model_name)
        logger.info(f"Saving {len(encoded_data)} encoded examples to: {output_path}")
        with open(output_path, "wb") as f_out:
            pickle.dump(encoded_data, f_out)
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.exception("An error occurred during processing")
        sys.exit(1)


if __name__ == "__main__":
    main()
