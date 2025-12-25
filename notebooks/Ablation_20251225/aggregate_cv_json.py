# 该脚本的目的是，对于clean_cv_json.py清洗后的JSON文件，
# 计算means和stds数组的均值，生成一个新的JSON文件，便于后续分析使用。
import json
from pathlib import Path
from typing import Dict, List


def average_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    """
    对 list[dict] 形式的指标求均值
    """
    if not items:
        return {}

    metrics = items[0].keys()
    avg = {}

    for m in metrics:
        avg[m] = sum(item[m] for item in items) / len(items)

    return avg


def aggregate_arrays(data: dict) -> dict:
    """
    遍历既定结构，对 means / stds 做均值聚合
    """
    for feature in data.values():              # SIF / SGF
        for model in feature.values():          # lr / rf / xgb
            for dataset in model.values():      # sif_sgf_second / US...
                for key in ("means", "stds"):
                    arr = dataset.get(key, [])
                    if isinstance(arr, list) and len(arr) > 0:
                        avg_item = average_metrics(arr)
                        dataset[key] = [avg_item]
    return data


def main():
    input_path = Path("all_cleaned.json")
    output_path = Path("all_averaged.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    averaged = aggregate_arrays(data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(averaged, f, indent=2, ensure_ascii=False)

    print(f"✅ Averaged JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
