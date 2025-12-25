# 该脚本的作用是，对于Phase3输出的all.json文件，清洗means和stds数组，
# 如果其中包含非法值（null，NaN，或者dict中包含NaN），就会将该数组替换为空数组 []。
# 使用后需要配合aggregate_cv_json.py,计算平均值

import json
import math
from pathlib import Path
from typing import Any


def is_valid_number(x: Any) -> bool:
    """
    判断一个值是否是合法数字（int / float，且不是 NaN）
    """
    if isinstance(x, (int, float)):
        return not math.isnan(x)
    return False


def is_invalid_item(item: Any) -> bool:
    """
    判断数组中的一个元素是否非法
    """
    # null
    if item is None:
        return True

    # dict：检查其所有 value
    if isinstance(item, dict):
        for v in item.values():
            if not is_valid_number(v):
                return True
        return False

    # 其他类型（比如 list / str），直接认为非法
    return True


def clean_array(arr: Any) -> list:
    """
    如果数组中存在非法元素，返回空数组 []
    否则原样返回
    """
    if not isinstance(arr, list):
        return []

    for item in arr:
        if is_invalid_item(item):
            return []

    return arr


def clean_cv_json(data: dict) -> dict:
    """
    按你的结构，递归清洗 means / stds
    """
    for feature in data.values():              # SIF / SGF
        for model in feature.values():          # lr / rf / xgb
            for dataset in model.values():      # sif_sgf_second / US...
                for key in ("means", "stds"):
                    dataset[key] = clean_array(dataset.get(key))
    return data


def main():
    input_path = Path("all.json")
    output_path = Path("all_cleaned.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = clean_cv_json(data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned JSON saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
