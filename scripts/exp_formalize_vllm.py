import os
import sys
from pathlib import Path
import json
import argparse
import random
import re
import time
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.common import compile_lean4, DEFAULT_LAKE_PATH, DEFAULT_LEAN_WORKSPACE


def extract_code_block(text):
    # Match triple backtick blocks, optionally with a language tag
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1)
        return code.strip()
    return None


def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def statement_translation_inference(informal_statement):
    return f"""
I want you to translate a informal statment to formal statement in lean 4, the informal statement of the problem is:

{informal_statement}

The output is
"""


def initilize_vllm(args):
    seed = int(time.time()) % 1000 + random.randint(0, 10) * 1000
    if args.quantized_model_path:
        model_path = args.quantized_model_path
    else:
        model_path = args.model_path
    llm = LLM(
        model=model_path,
        tokenizer=args.model_path,
        # max_num_batched_tokens=8192,
        seed=seed,
        trust_remote_code=True,
        max_model_len=2048,
        # cpu_offload_gb=20,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    return llm


def formalize(llm, informal_statement, n=1):
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
        n=n,
    )
    responses = llm.generate(
        [statement_translation_inference(informal_statement)],
        sampling_params=sampling_params,
    )
    return [extract_code_block(output.text) for output in responses[0].outputs]


def compile_statement(stmt):
    try:
        status, _ = compile_lean4(
            stmt, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE
        )
        return status
    except Exception as e:
        # print(f"Error validating {item['example_id']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quantized_model_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    args = parser.parse_args()

    data = load_data(args.input_path)
    # data = data[:5]
    llm = initilize_vllm(args)

    formalized_data = []
    for item in tqdm(data):
        preds = formalize(llm, item["informal_problem"], n=args.num_samples)
        for pred in preds:
            is_compiled = compile_statement(pred)
            formalized_data.append(
                item | {"formal_statement_pred": pred, "compiled": is_compiled}
            )

    with open(args.output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
