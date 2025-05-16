#!/usr/bin/env python3
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict

# Define buckets for lines of code and definitions as global variables
LOC_BUCKETS = [0, 10, 20, 30, 40, 50, float("inf")]
DEF_BUCKETS = [0, 1, 2, 3, 4, 5, float("inf")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate statistics on LEAN datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True, help="Paths to JSONL dataset files"
    )
    parser.add_argument("--output", default="stats_output", help="Output directory")
    return parser.parse_args()


def count_lines_and_defs(code_text):
    # Define definition keywords
    DEF_KEYWORDS = [
        "def",
        "structure",
        "class",
        "inductive",
        "coinductive",
        "abbrev",
        "instance",
        "mutual",
        "constant",
        "axiom",
    ]

    # Remove comments
    code_text = re.sub(r"--.*$", "", code_text, flags=re.MULTILINE)
    code_text = re.sub(r"/--.*?-/", "", code_text, flags=re.DOTALL)

    lines = code_text.split("\n")
    total_lines = 0
    definitions = 0

    # Track if we're in import/namespace section
    in_imports = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip imports and namespace declarations
        if line.startswith("import ") or line.startswith("open "):
            in_imports = True
            continue
        elif (
            in_imports
            and not line.startswith("import ")
            and not line.startswith("open ")
        ):
            in_imports = False

        if not in_imports:
            total_lines += 1

            # Check if line starts with any definition keyword
            for keyword in DEF_KEYWORDS:
                if line.startswith(keyword + " "):
                    # Extract the name of the definition
                    def_parts = line[len(keyword) :].strip().split(" ")
                    if def_parts:
                        def_name = def_parts[0]
                        # Only count if name doesn't contain "solution"
                        if "solution" not in def_name.lower():
                            definitions += 1
                    break

    return total_lines, definitions


def process_datasets(dataset_paths):
    # Helper function to determine bucket
    def get_bucket_label(value, buckets):
        for i in range(len(buckets) - 1):
            if buckets[i] <= value < buckets[i + 1]:
                if buckets[i + 1] == float("inf"):
                    return f"{buckets[i]}+"
                return f"{buckets[i]}-{buckets[i+1]}"
        return "unknown"

    results = {}
    # Add nl_text to highest examples tracking
    highest_examples = {
        "loc": {
            "count": 0,
            "problem": None,
            "code": None,
            "nl_text": None,
            "dataset": None,
        },
        "defs": {
            "count": 0,
            "problem": None,
            "code": None,
            "nl_text": None,
            "dataset": None,
        },
    }

    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        results[dataset_name] = {
            "loc": defaultdict(int),
            "defs": defaultdict(int),
            "max_loc_problem": {"count": 0, "problem": None},
            "max_defs_problem": {"count": 0, "problem": None},
        }

        # Process JSONL file
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "fl_text" not in data:
                        continue

                    code = data["fl_text"]
                    loc, defs = count_lines_and_defs(code)

                    # Track maximum problems
                    if loc > results[dataset_name]["max_loc_problem"]["count"]:
                        results[dataset_name]["max_loc_problem"] = {
                            "count": loc,
                            "problem": data.get("name", f"ID {line_num+1}"),
                        }
                    if defs > results[dataset_name]["max_defs_problem"]["count"]:
                        results[dataset_name]["max_defs_problem"] = {
                            "count": defs,
                            "problem": data.get("name", f"ID {line_num+1}"),
                        }

                    # Track global maximums with nl_text
                    if loc > highest_examples["loc"]["count"]:
                        highest_examples["loc"] = {
                            "count": loc,
                            "problem": data.get("name", f"ID {line_num+1}"),
                            "code": code,
                            "nl_text": data.get(
                                "nl_text", "No natural language text available"
                            ),
                            "dataset": dataset_name,
                        }
                    if defs > highest_examples["defs"]["count"]:
                        highest_examples["defs"] = {
                            "count": defs,
                            "problem": data.get("name", f"ID {line_num+1}"),
                            "code": code,
                            "nl_text": data.get(
                                "nl_text", "No natural language text available"
                            ),
                            "dataset": dataset_name,
                        }

                    # Assign to appropriate buckets
                    loc_bucket = get_bucket_label(loc, LOC_BUCKETS)
                    def_bucket = get_bucket_label(defs, DEF_BUCKETS)

                    # Increment counts in buckets
                    results[dataset_name]["loc"][loc_bucket] += 1
                    results[dataset_name]["defs"][def_bucket] += 1

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on ID {line_num+1} in {dataset_path}")
                except Exception as e:
                    print(f"Error processing ID {line_num+1} in {dataset_path}: {e}")

    return results, highest_examples


def generate_tables(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for LOC table
    loc_data = []
    for dataset, stats in results.items():
        for bucket, count in stats["loc"].items():
            loc_data.append({"Dataset": dataset, "LOC Bucket": bucket, "Count": count})

    # Prepare data for definitions table
    def_data = []
    for dataset, stats in results.items():
        for bucket, count in stats["defs"].items():
            def_data.append(
                {"Dataset": dataset, "Definitions Bucket": bucket, "Count": count}
            )

    # Create DataFrames
    loc_df = pd.DataFrame(loc_data)
    def_df = pd.DataFrame(def_data)

    # Sort by bucket values for better readability
    def bucket_sorter(bucket_str):
        return float(bucket_str.split("-")[0].replace("+", ""))

    loc_df["Bucket_Order"] = loc_df["LOC Bucket"].apply(bucket_sorter)
    def_df["Bucket_Order"] = def_df["Definitions Bucket"].apply(bucket_sorter)

    loc_df = loc_df.sort_values(["Dataset", "Bucket_Order"]).drop(
        "Bucket_Order", axis=1
    )
    def_df = def_df.sort_values(["Dataset", "Bucket_Order"]).drop(
        "Bucket_Order", axis=1
    )

    # Save tables
    loc_df.to_csv(os.path.join(output_dir, "lines_of_code.csv"), index=False)
    def_df.to_csv(os.path.join(output_dir, "definitions.csv"), index=False)

    return loc_df, def_df


def generate_figures(loc_df, def_df, output_dir):
    # Set style
    sns.set(style="whitegrid")

    # Lines of Code figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="LOC Bucket", y="Count", hue="Dataset", data=loc_df)
    chart.set_title("Distribution of Lines of Code")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lines_of_code.png"))

    # Definitions figure
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(x="Definitions Bucket", y="Count", hue="Dataset", data=def_df)
    chart.set_title("Distribution of Definitions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "definitions.png"))


def main():
    args = parse_args()
    results, highest_examples = process_datasets(args.datasets)
    loc_df, def_df = generate_tables(results, args.output)
    generate_figures(loc_df, def_df, args.output)

    print(f"Statistics saved to {args.output}")

    # Print highest examples with nl_text
    print("\nHighest Examples:")
    print("\nMost Lines of Code:")
    print(f"Dataset: {highest_examples['loc']['dataset']}")
    print(f"Problem: {highest_examples['loc']['problem']}")
    print(f"Lines of Code: {highest_examples['loc']['count']}")
    print("\nNatural Language Description:")
    print(highest_examples["loc"]["nl_text"])
    print("\nCode:")
    print(highest_examples["loc"]["code"])

    print("\nMost Definitions:")
    print(f"Dataset: {highest_examples['defs']['dataset']}")
    print(f"Problem: {highest_examples['defs']['problem']}")
    print(f"Number of Definitions: {highest_examples['defs']['count']}")
    print("\nNatural Language Description:")
    print(highest_examples["defs"]["nl_text"])
    print("\nCode:")
    print(highest_examples["defs"]["code"])

    # Print summary
    print("\nSummary by Dataset:")
    for dataset, buckets in results.items():
        total_loc = sum(buckets["loc"].values())
        total_defs = sum(buckets["defs"].values())

        # Get max/min/mean for LOC
        max_loc_count = max(buckets["loc"].values())
        max_loc_bucket = max(buckets["loc"].items(), key=lambda x: x[1])
        min_loc_count = min(buckets["loc"].values())
        min_loc_bucket = min(buckets["loc"].items(), key=lambda x: x[1])
        mean_loc = total_loc / len(buckets["loc"]) if buckets["loc"] else 0

        # Get max/min/mean for definitions
        max_def_count = max(buckets["defs"].values())
        max_def_bucket = max(buckets["defs"].items(), key=lambda x: x[1])
        min_def_count = min(buckets["defs"].values())
        min_def_bucket = min(buckets["defs"].items(), key=lambda x: x[1])
        mean_def = total_defs / len(buckets["defs"]) if buckets["defs"] else 0

        print(f"{dataset}:")
        print(f"  Total: {total_loc} lines of code, {total_defs} definitions")
        print(f"  Lines of code:")
        print(f"    Most common: {max_loc_bucket[0]} ({max_loc_count} examples)")
        print(f"    Least common: {min_loc_bucket[0]} ({min_loc_count} examples)")
        print(f"    Mean: {mean_loc:.1f} examples per bucket")
        print(f"  Definitions:")
        print(f"    Most common: {max_def_bucket[0]} ({max_def_count} examples)")
        print(f"    Least common: {min_def_bucket[0]} ({min_def_count} examples)")
        print(f"    Mean: {mean_def:.1f} examples per bucket")

        # Add maximum problem information
        print(f"  Largest problems:")
        print(
            f"    Most lines of code: {buckets['max_loc_problem']['problem']} "
            f"({buckets['max_loc_problem']['count']} lines)"
        )
        print(
            f"    Most definitions: {buckets['max_defs_problem']['problem']} "
            f"({buckets['max_defs_problem']['count']} definitions)"
        )


if __name__ == "__main__":
    main()
