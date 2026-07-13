# count_shrimp_report_table.py
# -*- coding: utf-8 -*-

import json
from pathlib import Path


REPORT_FILES = [
    {
        "資料類型": "影片描述",
        "資料切分": "訓練集",
        "file": "shrimp_video_cap_train_clean.json",
    },
    {
        "資料類型": "影片描述",
        "資料切分": "驗證集",
        "file": "shrimp_video_cap_val.json",
    },
    {
        "資料類型": "定位與計數",
        "資料切分": "訓練集",
        "file": "shrimp_video_point_train_final_clean.json",
    },
    {
        "資料類型": "定位與計數",
        "資料切分": "驗證集",
        "file": "shrimp_video_point_val_how_many_final.json",
    },
]


HF_FILES = {
    "hf_cap": "hf_cap.json",
    "hf_point": "hf_point.json",
    "hf_shrimp_train": "hf_shrimp_train.json",
}


def normalize_video_path(video_path: str) -> str:
    """
    將影片路徑統一成相對路徑，避免絕對路徑與相對路徑造成 unique video 統計錯誤。
    """
    if not video_path:
        return ""

    p = str(video_path).replace("\\", "/")

    marker = "/custom/shrimp/videos/"
    if marker in p:
        return p.split(marker, 1)[1]

    marker = "/shrimp/videos/"
    if marker in p:
        return p.split(marker, 1)[1]

    marker = "/videos/"
    if marker in p:
        return p.split(marker, 1)[1]

    return p


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_video(item: dict) -> str:
    return normalize_video_path(item.get("video", ""))


def count_flat_json(path: Path) -> dict:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(f"{path.name} 不是 list 格式")

    qa_count = len(data)
    videos = [get_video(item) for item in data if isinstance(item, dict) and get_video(item)]
    unique_videos = len(set(videos))
    avg = qa_count / unique_videos if unique_videos else 0

    return {
        "file": path.name,
        "問答筆數": qa_count,
        "影片數量": unique_videos,
        "平均每段影片問答數": round(avg, 2),
        "_videos_set": set(videos),
    }


def format_int(n: int) -> str:
    return f"{n:,}"


def print_markdown_table(rows: list):
    print("\n=== 報告用資料表 ===\n")
    print("| 資料類型 | 資料切分 | 問答筆數 | 影片數量 | 平均每段影片問答數 |")
    print("| ----- | ---- | -----: | ----: | --------: |")

    for r in rows:
        print(
            f"| {r['資料類型']} | {r['資料切分']} | "
            f"{format_int(r['問答筆數'])} | "
            f"{format_int(r['影片數量'])} | "
            f"{r['平均每段影片問答數']:.2f} |"
        )


def check_hf_files(base_dir: Path) -> dict:
    result = {}

    for key, filename in HF_FILES.items():
        path = base_dir / filename
        if path.exists():
            data = load_json(path)
            result[key] = {
                "file": filename,
                "exists": True,
                "count": len(data) if isinstance(data, list) else None,
            }
        else:
            result[key] = {
                "file": filename,
                "exists": False,
                "count": None,
            }

    hf_cap = result["hf_cap"]["count"] or 0
    hf_point = result["hf_point"]["count"] or 0
    hf_train = result["hf_shrimp_train"]["count"] or 0

    result["check"] = {
        "hf_cap_plus_hf_point": hf_cap + hf_point,
        "hf_shrimp_train": hf_train,
        "is_equal": (hf_cap + hf_point == hf_train),
    }

    return result


def main():
    base_dir = Path(".")
    print("目前資料夾：", base_dir.resolve())

    report_rows = []
    all_videos = set()

    for item in REPORT_FILES:
        path = base_dir / item["file"]

        if not path.exists():
            raise FileNotFoundError(f"找不到檔案：{item['file']}")

        stat = count_flat_json(path)

        row = {
            "資料類型": item["資料類型"],
            "資料切分": item["資料切分"],
            "來源檔案": item["file"],
            "問答筆數": stat["問答筆數"],
            "影片數量": stat["影片數量"],
            "平均每段影片問答數": stat["平均每段影片問答數"],
        }

        report_rows.append(row)
        all_videos |= stat["_videos_set"]

    total_qa = sum(r["問答筆數"] for r in report_rows)
    total_unique_videos = len(all_videos)

    print_markdown_table(report_rows)

    print("\n=== 總計 ===")
    print(f"總問答筆數：{format_int(total_qa)}")
    print(f"不重複影片數：{format_int(total_unique_videos)}")

    hf_check = check_hf_files(base_dir)

    print("\n=== LoRA 訓練資料檢查 ===")
    print(f"hf_cap.json：{format_int(hf_check['hf_cap']['count']) if hf_check['hf_cap']['exists'] else '找不到'}")
    print(f"hf_point.json：{format_int(hf_check['hf_point']['count']) if hf_check['hf_point']['exists'] else '找不到'}")
    print(f"hf_cap + hf_point：{format_int(hf_check['check']['hf_cap_plus_hf_point'])}")
    print(f"hf_shrimp_train.json：{format_int(hf_check['check']['hf_shrimp_train'])}")

    if hf_check["check"]["is_equal"]:
        print("檢查結果：OK，hf_shrimp_train.json = hf_cap.json + hf_point.json")
    else:
        print("檢查結果：注意，hf_shrimp_train.json 不等於 hf_cap.json + hf_point.json")

    output = {
        "report_table": report_rows,
        "total": {
            "問答筆數": total_qa,
            "不重複影片數": total_unique_videos,
        },
        "lora_training_check": hf_check,
    }

    out_path = base_dir / "shrimp_dataset_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n已輸出 JSON 檢查檔：{out_path}")


if __name__ == "__main__":
    main()