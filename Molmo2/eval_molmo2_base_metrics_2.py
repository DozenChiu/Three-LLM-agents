#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception as e:
    print("[ERROR] transformers import failed:", e)
    raise


# ============================================================
# Regex / parsing
# ============================================================

# Your shrimp GT format:
# <point x="0.08" y="0.35" t="0.0"></point>
SHRIMP_POINT_TAG_RE = re.compile(
    r"<point\s+([^>]*?)\s*/?>.*?(?:</point>)?",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r'(x|y|t)\s*=\s*"([^"]+)"', re.IGNORECASE)

# Molmo2 native format:
# <points coords="0.0 1 80 350;0.5 2 520 480">shrimp</points>
MOLMO_POINTS_RE = re.compile(
    r"<points\s+[^>]*?coords\s*=\s*\"([^\"]+)\"",
    re.IGNORECASE | re.DOTALL,
)

# Optional fallback if the model outputs tracks.
MOLMO_TRACKS_RE = re.compile(
    r"<tracks\s+[^>]*?coords\s*=\s*\"([^\"]+)\"",
    re.IGNORECASE | re.DOTALL,
)

# Count text patterns.
DIGIT_COUNT_RE = re.compile(
    r"\b(?:there\s+(?:is|are)\s+|I\s+(?:can\s+)?see\s+|counting\s+.*?shows\s+(?:a\s+total\s+of\s+)?)"
    r"(\d+)\s+(?:shrimp|shrimps)\b",
    re.IGNORECASE | re.DOTALL,
)

ANY_DIGIT_SHRIMP_RE = re.compile(
    r"\b(\d+)\s+(?:shrimp|shrimps)\b",
    re.IGNORECASE,
)

INTEGER_ONLY_RE = re.compile(r"^\s*(\d+)\s*[\.\)]?\s*$")

WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

WORD_COUNT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(WORD_NUMBERS.keys(), key=len, reverse=True)) + r")\s+(?:shrimp|shrimps)\b",
    re.IGNORECASE,
)


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def extract_qa(example: Dict[str, Any]) -> Tuple[str, str]:
    conv = example["conversations"]
    question = next(x["value"] for x in conv if x["from"] == "human")
    answer = next(x["value"] for x in conv if x["from"] == "gpt")
    return question, answer


# ============================================================
# Point parsing
# ============================================================

def parse_shrimp_gt_points(text: str) -> List[Tuple[float, float, float]]:
    """
    Parse your shrimp GT format:
    <point x="0.08" y="0.35" t="0.0"></point>

    Return:
    [(x_0_to_1, y_0_to_1, t), ...]
    """
    points: List[Tuple[float, float, float]] = []

    for tag_match in SHRIMP_POINT_TAG_RE.finditer(text or ""):
        attrs_text = tag_match.group(1)
        attrs: Dict[str, float] = {}

        for key, value in ATTR_RE.findall(attrs_text):
            try:
                attrs[key.lower()] = float(value)
            except ValueError:
                continue

        if "x" in attrs and "y" in attrs:
            points.append((attrs["x"], attrs["y"], attrs.get("t", 0.0)))

    return points


def _parse_molmo_coords_string(
    coords: str,
    video_duration: Optional[float] = None,
    unique_object_only: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Parse Molmo2 native coords string.

    Expected video-style group:
      timestamp object_id x y object_id x y ...

    Example:
      0.0 1 80 350;0.5 2 520 480

    Molmo2 x/y are normalized integers in 0~1000.
    We convert them to 0~1.

    For <tracks>, the same object_id may appear across many timestamps.
    If unique_object_only=True, keep only the first point for each object_id.
    This makes tracks comparable with the shrimp point/count validation set.
    """
    points: List[Tuple[float, float, float]] = []
    seen_object_ids = set()

    if not coords:
        return points

    chunks = re.split(r"[;\n\t]+", coords.strip())

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        nums = re.findall(r"-?\d+(?:\.\d+)?", chunk)
        if len(nums) < 4:
            continue

        try:
            time_or_index = float(nums[0])
        except ValueError:
            continue

        if video_duration and video_duration > 0:
            t_value = max(0.0, min(1.0, time_or_index / video_duration))
        else:
            t_value = time_or_index

        rest = nums[1:]

        for i in range(0, len(rest) - 2, 3):
            try:
                obj_id = int(float(rest[i]))
                x_raw = float(rest[i + 1])
                y_raw = float(rest[i + 2])
            except ValueError:
                continue

            if unique_object_only and obj_id in seen_object_ids:
                continue
            seen_object_ids.add(obj_id)

            # 【修改】對齊 Gradio 的尺度轉換邏輯
            if x_raw > 100 or y_raw > 100:
                x = x_raw / 1000.0
                y = y_raw / 1000.0
            elif x_raw > 1 or y_raw > 1:
                x = x_raw / 100.0
                y = y_raw / 100.0
            else:
                x = x_raw
                y = y_raw

            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            points.append((x, y, t_value))

    return points


def parse_molmo_native_points(text: str, video_duration: Optional[float] = None, include_tracks: bool = False) -> List[Tuple[float, float, float]]:
    """
    Parse Molmo2 native outputs:
      <points coords="...">shrimp</points>

    Optional:
      <tracks coords="...">shrimp</tracks>
    """
    points: List[Tuple[float, float, float]] = []

    for m in MOLMO_POINTS_RE.finditer(text or ""):
        coords = m.group(1)
        points.extend(_parse_molmo_coords_string(coords, video_duration=video_duration))

    if include_tracks:
        for m in MOLMO_TRACKS_RE.finditer(text or ""):
            coords = m.group(1)
            points.extend(_parse_molmo_coords_string(coords, video_duration=video_duration))

    return points


def parse_count_from_text(text: str) -> Optional[int]:
    """
    Robust text count parser for base model outputs.
    Handles:
      "There are 5 shrimps"
      "I can see five shrimp"
      "5"
      "Counting ... shows a total of 10"
    """
    s = normalize_text(text)

    m = INTEGER_ONLY_RE.search(s)
    if m:
        return int(m.group(1))

    m = DIGIT_COUNT_RE.search(s)
    if m:
        return int(m.group(1))

    m = ANY_DIGIT_SHRIMP_RE.search(s)
    if m:
        return int(m.group(1))

    m = WORD_COUNT_RE.search(s)
    if m:
        return WORD_NUMBERS.get(m.group(1).lower())

    return None


def count_from_molmo_native_points(text: str, video_duration: Optional[float] = None) -> Optional[int]:
    pts = parse_molmo_native_points(text, video_duration=video_duration)
    if pts:
        return len(pts)
    return None


# ============================================================
# Metrics helpers
# ============================================================

def point_distance(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    include_time: bool,
) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    if include_time:
        dt = p1[2] - p2[2]
        return math.sqrt(dx * dx + dy * dy + dt * dt)
    return math.sqrt(dx * dx + dy * dy)


def greedy_assignment(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Hungarian if scipy exists; otherwise greedy fallback.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        pairs: List[Tuple[int, int]] = []
        used_rows = set()
        used_cols = set()
        flat: List[Tuple[float, int, int]] = []

        for r, row in enumerate(cost):
            for c, v in enumerate(row):
                flat.append((v, r, c))

        flat.sort(key=lambda x: x[0])

        for _, r, c in flat:
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))

        return pairs


def close_count_correct(pred: int, gt: int) -> bool:
    """
    Close accuracy similar to Molmo2 counting benchmark:
    correct if |pred - gt| <= 1 + floor(0.05 * gt)
    """
    delta = 1 + math.floor(0.05 * gt)
    return abs(pred - gt) <= delta


# ============================================================
# Model runner
# ============================================================

class MolmoBaseRunner:
    def __init__(
        self,
        base_model_path: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
        do_sample: bool,
        top_p: float,
        temperature: float,
    ) -> None:
        self.base_model_path = base_model_path
        self.device = device
        self.dtype = self._resolve_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature

        print("[INFO] Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )
        print("[INFO] Processor loaded.")

        print("[INFO] Loading base Molmo2 model only...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()
        print("[INFO] Base model loaded.")

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        dtype = dtype.lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    def generate_answer(self, video_path: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt_text],
            videos=[video_path],
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }

        if self.do_sample:
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, input_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return text


# ============================================================
# Prompting
# ============================================================

def build_caption_prompt(question: str) -> str:
    return question


def build_point_prompt(question: str, label: str = "shrimp") -> str:
    """
    Prompt base Molmo2-4B to use Molmo2 native point format.
    """
    return (
        f"{question}\n\n"
        f"Answer by pointing to every relevant {label} in the video using Molmo2 native point format.\n"
        f"Return only this format:\n"
        f"<points coords=\"timestamp object_id x y; timestamp object_id x y\">{label}</points>\n"
        f"Rules:\n"
        f"- timestamp should be in seconds, such as 0.0, 0.5, 1.0.\n"
        f"- object_id starts from 1 and increases by 1 for each distinct {label}.\n"
        f"- x and y are integer coordinates normalized from 0 to 1000.\n"
        f"- Do not add explanation outside the tag."
    )


# ============================================================
# Inference
# ============================================================

def run_inference(
    runner: MolmoBaseRunner,
    dataset: Sequence[Dict[str, Any]],
    video_root: Path,
    out_jsonl: Path,
    task: str,
    max_samples: Optional[int] = None,
    resume: bool = True,
) -> List[Dict[str, Any]]:
    existing_rows = load_jsonl(out_jsonl) if resume else []
    done_ids = {str(x.get("id")) for x in existing_rows}

    if not resume and out_jsonl.exists():
        out_jsonl.unlink()

    rows: List[Dict[str, Any]] = list(existing_rows)
    subset = dataset[:max_samples] if max_samples is not None else dataset

    for ex in tqdm(subset, desc=f"Infer {out_jsonl.stem}"):
        ex_id = str(ex.get("id"))

        if ex_id in done_ids:
            continue

        question, reference = extract_qa(ex)
        video_path = video_root / ex["video"]

        if task == "point":
            prompt = build_point_prompt(question)
        else:
            prompt = build_caption_prompt(question)

        row: Dict[str, Any] = {
            "id": ex.get("id"),
            "video": ex["video"],
            "task": task,
            "question": question,
            "prompt_used": prompt,
            "reference": reference,
            "prediction": "",
            "ok": False,
            "error": None,
        }

        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            pred = runner.generate_answer(str(video_path), prompt)
            row["prediction"] = pred
            row["ok"] = True

        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[WARN] id={ex_id} failed: {row['error']}")

        append_jsonl(out_jsonl, row)
        rows.append(row)

    return rows


# ============================================================
# Caption metrics
# ============================================================

def compute_caption_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [
        x for x in rows
        if x.get("ok") and normalize_text(x.get("prediction", ""))
    ]

    predictions = [normalize_text(x["prediction"]) for x in ok_rows]
    references = [normalize_text(x["reference"]) for x in ok_rows]

    if not predictions:
        return {
            "num_samples": 0,
            "BLEU-1": None,
            "BLEU-2": None,
            "ROUGE-1 F1": None,
            "ROUGE-2 F1": None,
            "ROUGE-L F1": None,
            "METEOR": None,
            "BERTScore F1": None,
        }

    try:
        from rouge_score import rouge_scorer
        import nltk
        from nltk.translate.meteor_score import meteor_score
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from bert_score import score as bertscore_score

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        cc = SmoothingFunction()
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )

        bleu1_list = []
        bleu2_list = []
        rouge1_list = []
        rouge2_list = []
        rougeL_list = []
        meteor_list = []

        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred)
            ref_tokens = nltk.word_tokenize(ref)

            bleu1_list.append(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(1.0, 0.0, 0.0, 0.0),
                    smoothing_function=cc.method1,
                )
            )

            bleu2_list.append(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.5, 0.5, 0.0, 0.0),
                    smoothing_function=cc.method1,
                )
            )

            scores = scorer.score(ref, pred)
            rouge1_list.append(scores["rouge1"].fmeasure)
            rouge2_list.append(scores["rouge2"].fmeasure)
            rougeL_list.append(scores["rougeL"].fmeasure)

            meteor_list.append(meteor_score([ref_tokens], pred_tokens))

        _, _, F1 = bertscore_score(predictions, references, lang="en", verbose=False)

        return {
            "num_samples": len(ok_rows),
            "BLEU-1": round(float(sum(bleu1_list) / len(bleu1_list)), 4),
            "BLEU-2": round(float(sum(bleu2_list) / len(bleu2_list)), 4),
            "ROUGE-1 F1": round(float(sum(rouge1_list) / len(rouge1_list)), 4),
            "ROUGE-2 F1": round(float(sum(rouge2_list) / len(rouge2_list)), 4),
            "ROUGE-L F1": round(float(sum(rougeL_list) / len(rougeL_list)), 4),
            "METEOR": round(float(sum(meteor_list) / len(meteor_list)), 4),
            "BERTScore F1": round(float(F1.mean()), 4),
        }

    except Exception as e:
        return {
            "num_samples": len(ok_rows),
            "error": f"Caption metric computation failed: {type(e).__name__}: {e}",
            "hint": "Please install: pip install rouge-score nltk bert-score",
        }


# ============================================================
# Base point + text counting metrics
# ============================================================

def compute_base_point_metrics(
    rows: Sequence[Dict[str, Any]],
    include_time: bool = False,
    video_duration: Optional[float] = None,
) -> Dict[str, Any]:
    ok_rows = [x for x in rows if x.get("ok")]

    point_count_correct = 0
    point_count_abs_errors: List[float] = []

    text_count_correct = 0
    text_count_close_correct = 0
    text_count_abs_errors: List[float] = []
    text_count_total = 0

    native_parse_success = 0

    total_gt_points = 0
    matched_distances: List[float] = []
    hits_5 = 0
    hits_10 = 0

    per_sample_debug = []

    for row in ok_rows:
        reference = row.get("reference", "")
        prediction = row.get("prediction", "")

        gt_points = parse_shrimp_gt_points(reference)
        pred_points = parse_molmo_native_points(
            prediction,
            video_duration=video_duration,
            include_tracks=True,
        )
        # 【新增】將單點格式 <point x="..." y="..."> 的解析結果也加進來
        single_points = parse_shrimp_gt_points(prediction)
        pred_points.extend(single_points)
        
        # 【新增】去除重複的座標，避免兩個正規表達式都抓到同一個點導致重複計算
        pred_points = list(set(pred_points))

        gt_count = len(gt_points)
        pred_point_count = len(pred_points)

        total_gt_points += gt_count

        if pred_point_count > 0:
            native_parse_success += 1

        if pred_point_count == gt_count:
            point_count_correct += 1

        point_count_abs_errors.append(abs(pred_point_count - gt_count))

        gt_text_count = parse_count_from_text(reference)
        if gt_text_count is None:
            gt_text_count = gt_count

        pred_text_count = parse_count_from_text(prediction)

        if pred_text_count is not None:
            text_count_total += 1

            if pred_text_count == gt_text_count:
                text_count_correct += 1

            if close_count_correct(pred_text_count, gt_text_count):
                text_count_close_correct += 1

            text_count_abs_errors.append(abs(pred_text_count - gt_text_count))

        if gt_points and pred_points:
            cost = [
                [point_distance(g, p, include_time=include_time) for p in pred_points]
                for g in gt_points
            ]

            for r, c in greedy_assignment(cost):
                d = cost[r][c]
                matched_distances.append(d)

                if d <= 0.05:
                    hits_5 += 1
                if d <= 0.10:
                    hits_10 += 1

        per_sample_debug.append(
            {
                "id": row.get("id"),
                "gt_point_count": gt_count,
                "pred_native_point_count": pred_point_count,
                "gt_text_count": gt_text_count,
                "pred_text_count": pred_text_count,
                "prediction": prediction,
            }
        )

    return {
        "num_samples": len(ok_rows),

        "Native Point Parse Rate": round(native_parse_success / len(ok_rows), 4) if ok_rows else None,

        "Native Point Count Accuracy": round(point_count_correct / len(ok_rows), 4) if ok_rows else None,
        "Native Point Count MAE": round(sum(point_count_abs_errors) / len(point_count_abs_errors), 4) if point_count_abs_errors else None,

        "Text Count Parsed Samples": text_count_total,
        "Text Count Parse Rate": round(text_count_total / len(ok_rows), 4) if ok_rows else None,
        "Text Count Accuracy": round(text_count_correct / text_count_total, 4) if text_count_total else None,
        "Text Count Close Accuracy": round(text_count_close_correct / text_count_total, 4) if text_count_total else None,
        "Text Count MAE": round(sum(text_count_abs_errors) / len(text_count_abs_errors), 4) if text_count_abs_errors else None,

        "Average Coordinate Error": round(sum(matched_distances) / len(matched_distances), 4) if matched_distances else None,
        "Hit Rate (5%)": round(hits_5 / total_gt_points, 4) if total_gt_points else None,
        "Hit Rate (10%)": round(hits_10 / total_gt_points, 4) if total_gt_points else None,

        "total_gt_points": total_gt_points,
        "matched_pairs": len(matched_distances),
        "include_time_in_distance": include_time,
        "video_duration_for_time_normalization": video_duration,
    }


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Base Molmo2-4B eval for shrimp caption, text counting, and native Molmo2 point coordinates."
    )

    p.add_argument(
        "--base-model-path",
        required=True,
        help="HF Molmo2 base model path, e.g. /media/sdb/dozen/models/molmo2-4b",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Root folder containing shrimp json files. Default: $MOLMO_DATA_DIR/custom/shrimp",
    )
    p.add_argument(
        "--video-root",
        default=None,
        help="Video root folder. Default: <data-root>/videos",
    )
    p.add_argument(
        "--cap-val-json",
        default=None,
        help="Caption validation json path",
    )
    p.add_argument(
        "--point-val-json",
        default=None,
        help="Point validation json path",
    )
    p.add_argument(
        "--output-dir",
        default="./eval_outputs/base_4b_native",
        help="Directory for prediction jsonl and summary json",
    )
    p.add_argument(
        "--mode",
        choices=["cap", "point", "both"],
        default="both",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )
    p.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
    )
    p.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. Off by default for deterministic eval.",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subset for smoke test.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing jsonl predictions.",
    )
    p.add_argument(
        "--include-time-in-point-error",
        action="store_true",
        help="Include t in point distance. Usually keep this off for this project.",
    )
    p.add_argument(
        "--video-duration",
        type=float,
        default=None,
        help="Optional duration in seconds for converting Molmo2 timestamp seconds into 0~1. Usually leave unset unless using include-time.",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    molmo_data_dir = os.environ.get("MOLMO_DATA_DIR")
    default_data_root = Path(molmo_data_dir) / "custom" / "shrimp" if molmo_data_dir else None

    data_root = Path(args.data_root) if args.data_root else default_data_root
    if data_root is None:
        raise RuntimeError("--data-root not provided and MOLMO_DATA_DIR is not set.")

    video_root = Path(args.video_root) if args.video_root else data_root / "videos"
    cap_val_json = Path(args.cap_val_json) if args.cap_val_json else data_root / "shrimp_video_cap_val.json"
    point_val_json = Path(args.point_val_json) if args.point_val_json else data_root / "shrimp_video_point_val_how_many_final.json"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] data_root:", data_root)
    print("[INFO] video_root:", video_root)
    print("[INFO] cap_val_json:", cap_val_json)
    print("[INFO] point_val_json:", point_val_json)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] Base eval only. No LoRA will be loaded.")

    runner = MolmoBaseRunner(
        base_model_path=args.base_model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    summary: Dict[str, Any] = {
        "config": {
            "base_model_path": args.base_model_path,
            "lora_path": None,
            "data_root": str(data_root),
            "video_root": str(video_root),
            "cap_val_json": str(cap_val_json),
            "point_val_json": str(point_val_json),
            "output_dir": str(output_dir),
            "mode": args.mode,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "max_samples": args.max_samples,
            "resume": not args.no_resume,
            "include_time_in_point_error": args.include_time_in_point_error,
            "video_duration": args.video_duration,
            "point_prediction_format": "Molmo2 native <points coords=\"timestamp object_id x y\"> format",
            "gt_point_format": "shrimp <point x=\"...\" y=\"...\" t=\"...\"></point> format",
        }
    }

    try:
        if args.mode in {"cap", "both"}:
            cap_data = load_json(cap_val_json)
            cap_jsonl = output_dir / "cap_predictions.jsonl"

            cap_rows = run_inference(
                runner=runner,
                dataset=cap_data,
                video_root=video_root,
                out_jsonl=cap_jsonl,
                task="cap",
                max_samples=args.max_samples,
                resume=not args.no_resume,
            )

            summary["caption_metrics"] = compute_caption_metrics(cap_rows)

        if args.mode in {"point", "both"}:
            point_data = load_json(point_val_json)
            point_jsonl = output_dir / "point_predictions_native.jsonl"

            point_rows = run_inference(
                runner=runner,
                dataset=point_data,
                video_root=video_root,
                out_jsonl=point_jsonl,
                task="point",
                max_samples=args.max_samples,
                resume=not args.no_resume,
            )

            summary["point_metrics"] = compute_base_point_metrics(
                point_rows,
                include_time=args.include_time_in_point_error,
                video_duration=args.video_duration,
            )

    finally:
        clear_gpu()

    dump_json(output_dir / "metrics_summary.json", summary)

    print("\n================ SUMMARY ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=========================================")
    print(f"[DONE] Saved summary to: {output_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)