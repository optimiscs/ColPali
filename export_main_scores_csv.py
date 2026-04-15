from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULTS_DIR = Path("/home/moxu/MMRAG/otherExp/colpali/mteb_results/results")
DEFAULT_OUTPUT_CSV = Path("/home/moxu/MMRAG/otherExp/colpali/mteb_results/main_scores_by_category.csv")

VERSION_SUFFIX_RE = re.compile(r"\.v\d+$")
VIDORE_PREFIX_RE = re.compile(r"^Vidore\d+")
RETRIEVAL_SUFFIX_RE = re.compile(r"Retrieval$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 MTEB 结果目录中各模型的 main_score，并按任务类别输出到统一 CSV。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="MTEB results 根目录，默认是 ./mteb_results/results",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="输出 CSV 路径。",
    )
    return parser.parse_args()


def infer_category(task_name: str) -> str:
    base = VERSION_SUFFIX_RE.sub("", task_name)
    base = VIDORE_PREFIX_RE.sub("", base)
    base = RETRIEVAL_SUFFIX_RE.sub("", base)
    return base or task_name


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_score(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.5f}"


def build_model_labels(results_dir: Path) -> dict[tuple[str, str], str]:
    revisions_by_model: dict[str, set[str]] = defaultdict(set)
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for revision_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            revisions_by_model[model_dir.name].add(revision_dir.name)

    labels: dict[tuple[str, str], str] = {}
    for model_name, revisions in revisions_by_model.items():
        for revision in revisions:
            if len(revisions) == 1:
                labels[(model_name, revision)] = model_name
            else:
                labels[(model_name, revision)] = f"{model_name}@{revision}"
    return labels


def collect_rows(results_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    model_labels = build_model_labels(results_dir)
    model_columns = sorted(model_labels.values())
    rows_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        for revision_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            model_column = model_labels[(model_dir.name, revision_dir.name)]
            for json_path in sorted(revision_dir.glob("*.json")):
                data = load_json(json_path)
                task_name = data.get("task_name", json_path.stem)
                category = infer_category(task_name)
                scores = data.get("scores", {})

                for split_name, split_entries in scores.items():
                    for entry in split_entries:
                        hf_subset = entry.get("hf_subset", "")
                        row_key = (category, task_name, split_name, hf_subset)
                        row = rows_by_key.setdefault(
                            row_key,
                            {
                                "category": category,
                                "task_name": task_name,
                                "split": split_name,
                                "hf_subset": hf_subset,
                                "languages": ",".join(entry.get("languages", [])),
                            },
                        )
                        row[model_column] = format_score(entry.get("main_score"))

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    return rows, model_columns


def write_csv(output_csv: Path, rows: list[dict[str, str]], model_columns: list[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "task_name", "split", "hf_subset", "languages", *model_columns]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows, model_columns = collect_rows(args.results_dir)
    write_csv(args.output_csv, rows, model_columns)
    print(f"Wrote {len(rows)} rows for {len(model_columns)} models to {args.output_csv}")


if __name__ == "__main__":
    main()
