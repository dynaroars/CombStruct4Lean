import argparse
import json
from collections import defaultdict
from pathlib import Path


def process_jsonl_file(file_path):
    # Initialize counters
    compiled_count = 0
    aligned_count = 0

    # Track processed IDs to avoid double counting
    id_status = defaultdict(lambda: {"compiled": False, "aligned": False})

    try:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    item_id = item.get("id")

                    if item_id is None:
                        continue

                    # Update compiled status
                    if item.get("compiled") and not id_status[item_id]["compiled"]:
                        compiled_count += 1
                        id_status[item_id]["compiled"] = True

                    # Update aligned status
                    if item.get("aligned") and not id_status[item_id]["aligned"]:
                        aligned_count += 1
                        id_status[item_id]["aligned"] = True

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}")
                    continue

        return compiled_count, aligned_count

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL files and count compiled/aligned items"
    )
    parser.add_argument("files", nargs="+", help="Path to JSONL files")

    args = parser.parse_args()

    for file_path in args.files:
        path = Path(file_path)
        compiled_count, aligned_count = process_jsonl_file(path)
        print(f"\nResults for {path}:")
        print(f"Compiled items: {compiled_count}")
        print(f"Aligned items: {aligned_count}")


if __name__ == "__main__":
    main()
