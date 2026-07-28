"""Validate processed dataset files without calling LLMs."""
import argparse
import json
import config


def count_jsonl(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Check MAGER dataset files")
    parser.add_argument("--dataset", choices=sorted(config.DATASETS.keys()), default="politifact")
    args = parser.parse_args()

    config.set_dataset(args.dataset)
    for name, path in config.DATA_PATHS.items():
        print(f"{name}: {path}")
        print(f"  records: {count_jsonl(path)}")


if __name__ == "__main__":
    main()
