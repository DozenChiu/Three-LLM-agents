#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except Exception as e:  # pragma: no cover
    print("[ERROR] transformers import failed:", e)
    raise

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


# -----------------------------
# Helpers
# -----------------------------

POINT_TAG_RE = re.compile(
    r"<point\s+([^>]*?)\s*/?>.*?(?:</point>)?",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r'(x|y|t)\s*=\s*"([^\"]+)"', re.IGNORECASE)
COUNT_RE = re.compile(r"\bThere\s+(?:is|are)\s+(\d+)\s+shrimp", re.IGNORECASE)


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
    rows: List[Dict[str, Any]] = []
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


def parse_points(text: str) -> List[Tuple[float, float, float]]:
    """Parse <point x="..." y="..." t="..."></point> tags.

    The regex is intentionally forgiving because some generations can be malformed.
    Missing t defaults to 0.0.
    """
    points: List[Tuple[float, float, float]] = []
    for tag_match in POINT_TAG_RE.finditer(text or ""):
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


def parse_count_from_text(text: str) -> Optional[int]:
    m = COUNT_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def greedy_assignment(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Hungarian if SciPy exists, otherwise greedy fallback."""
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


@dataclass
class PointEvalResult:
    point_count_accuracy: float
    point_count_mae: float
    average_coordinate_error: float
    hit_rate_5: float
    hit_rate_10: float
    num_samples: int
    total_gt_points: int
    matched_pairs: int


# -----------------------------
# Model wrapper
# -----------------------------

class MolmoRunner:
    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[str],
        device: str,
        dtype: str,
        max_new_tokens: int,
        do_sample: bool,
        top_p: float,
        temperature: float,
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_path = lora_path
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

        print("[INFO] Loading base model...")
        model = AutoModelForImageTextToText.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=self.device,
        )
        print("[INFO] Base model loaded.")

        if self.lora_path:
            if PeftModel is None:
                raise RuntimeError("peft is not installed, but --lora-path was provided.")
            print(f"[INFO] Loading LoRA from: {self.lora_path}")
            model = PeftModel.from_pretrained(model, self.lora_path)
            print("[INFO] LoRA loaded.")

        model.eval()
        self.model = model

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

    def generate_answer(self, video_path: str, question: str) -> str:
        # This follows the HF inference pattern already validated in the prior project PDF.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
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


# -----------------------------
# Inference
# -----------------------------

def run_inference(
    runner: MolmoRunner,
    dataset: Sequence[Dict[str, Any]],
    video_root: Path,
    out_jsonl: Path,
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
        row: Dict[str, Any] = {
            "id": ex.get("id"),
            "video": ex["video"],
            "question": question,
            "reference": reference,
            "prediction": "",
            "ok": False,
            "error": None,
        }
        try:
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            pred = runner.generate_answer(str(video_path), question)
            row["prediction"] = pred
            row["ok"] = True
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[WARN] id={ex_id} failed: {row['error']}")
        append_jsonl(out_jsonl, row)
        rows.append(row)
    return rows


# -----------------------------
# Caption metrics
# -----------------------------

def compute_caption_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [x for x in rows if x.get("ok") and normalize_text(x.get("prediction", ""))]
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

        #  必要資源（避免 runtime error）
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        cc = SmoothingFunction()

        bleu1_list = []
        bleu2_list = []
        rouge1_list = []
        rouge2_list = []
        rougeL_list = []
        meteor_list = []

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred)
            ref_tokens = nltk.word_tokenize(ref)

            # BLEU-1
            bleu1_list.append(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(1.0, 0.0, 0.0, 0.0),
                    smoothing_function=cc.method1,
                )
            )

            # BLEU-2
            bleu2_list.append(
                sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(0.5, 0.5, 0.0, 0.0),
                    smoothing_function=cc.method1,
                )
            )

            # ROUGE
            scores = scorer.score(ref, pred)
            rouge1_list.append(scores["rouge1"].fmeasure)
            rouge2_list.append(scores["rouge2"].fmeasure)
            rougeL_list.append(scores["rougeL"].fmeasure)

            # METEOR
            meteor_list.append(meteor_score([ref_tokens], pred_tokens))

        bleu1_score = sum(bleu1_list) / len(bleu1_list)
        bleu2_score = sum(bleu2_list) / len(bleu2_list)
        rouge1 = sum(rouge1_list) / len(rouge1_list)
        rouge2 = sum(rouge2_list) / len(rouge2_list)
        rougeL = sum(rougeL_list) / len(rougeL_list)
        meteor_avg = sum(meteor_list) / len(meteor_list)

        # BERTScore
        _, _, F1 = bertscore_score(predictions, references, lang="en", verbose=False)
        bert_f1 = float(F1.mean())

        return {
            "num_samples": len(ok_rows),
            "BLEU-1": round(float(bleu1_score), 4),
            "BLEU-2": round(float(bleu2_score), 4),
            "ROUGE-1 F1": round(float(rouge1), 4),
            "ROUGE-2 F1": round(float(rouge2), 4),
            "ROUGE-L F1": round(float(rougeL), 4),
            "METEOR": round(float(meteor_avg), 4),
            "BERTScore F1": round(float(bert_f1), 4),
        }

    except Exception as e:
        return {
            "num_samples": len(ok_rows),
            "error": f"Caption metric computation failed: {type(e).__name__}: {e}",
            "hint": "Please install: pip install rouge-score nltk bert-score",
        }


# -----------------------------
# Point metrics
# -----------------------------

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


def compute_point_metrics(rows: Sequence[Dict[str, Any]], include_time: bool = False) -> Dict[str, Any]:
    ok_rows = [x for x in rows if x.get("ok")]
    exact_count_correct = 0
    count_abs_errors: List[float] = []
    matched_distances: List[float] = []
    total_gt_points = 0
    hits_5 = 0
    hits_10 = 0
    count_text_match = 0
    count_text_total = 0

    for row in ok_rows:
        # ...解析點位...
        gt_points = parse_points(row.get("reference", ""))
        pred_points = parse_points(row.get("prediction", ""))

        gt_count = len(gt_points)
        pred_count = len(pred_points)
        total_gt_points += gt_count     # 累加分母 G

        if pred_count == gt_count:
            exact_count_correct += 1
        count_abs_errors.append(abs(pred_count - gt_count))

        gt_count_text = parse_count_from_text(row.get("reference", ""))
        pred_count_text = parse_count_from_text(row.get("prediction", ""))
        if gt_count_text is not None and pred_count_text is not None:
            count_text_total += 1
            if gt_count_text == pred_count_text:
                count_text_match += 1

        if gt_points and pred_points:
            # 建立距離成本矩陣
            cost = [
                [point_distance(g, p, include_time=include_time) for p in pred_points]
                for g in gt_points
            ]

            # 進行一對一配對 (取得配對集合 A*)
            for r, c in greedy_assignment(cost):
                d = cost[r][c]  # 這就是配對後的 d_ij
                matched_distances.append(d) # 收集以計算 Ecoord

                # 統計 N_0.05 與 N_0.10
                if d <= 0.05:
                    hits_5 += 1
                if d <= 0.10:
                    hits_10 += 1

    out = {
        "num_samples": len(ok_rows),
        "Point Count Accuracy": round(exact_count_correct / len(ok_rows), 4) if ok_rows else None,
        "Point Count MAE": round(sum(count_abs_errors) / len(count_abs_errors), 4) if count_abs_errors else None,
        # 平均座標誤差 (E_coord) = 距離總和 / 配對點數 (M)
        "Average Coordinate Error": round(sum(matched_distances) / len(matched_distances), 4) if matched_distances else None,
        # Hit Rate 5% (HR_0.05) = hits_5 (N_0.05) / total_gt_points (G)
        "Hit Rate (5%)": round(hits_5 / total_gt_points, 4) if total_gt_points else None,
        # Hit Rate 10% (HR_0.10) = hits_10 (N_0.10) / total_gt_points (G)
        "Hit Rate (10%)": round(hits_10 / total_gt_points, 4) if total_gt_points else None,
        "total_gt_points": total_gt_points,
        "matched_pairs": len(matched_distances),
        "count_text_accuracy_debug": round(count_text_match / count_text_total, 4) if count_text_total else None,
        "include_time_in_distance": include_time,
    }
    return out


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference + evaluation for shrimp caption and point validation sets.")
    p.add_argument("--base-model-path", required=True, help="HF Molmo2 base model path, e.g. /media/sdb/dozen/models/molmo2-4b")
    p.add_argument("--lora-path", default=None, help="Optional PEFT LoRA checkpoint path")
    p.add_argument("--data-root", default=None, help="Root folder containing shrimp json files. Default: $MOLMO_DATA_DIR/custom/shrimp")
    p.add_argument("--video-root", default=None, help="Video root folder. Default: <data-root>/videos")
    p.add_argument("--cap-val-json", default=None, help="Caption validation json path")
    p.add_argument("--point-val-json", default=None, help="Point validation json path")
    p.add_argument("--output-dir", default="./eval_outputs", help="Directory for prediction jsonl and summary json")
    p.add_argument("--mode", choices=["cap", "point", "both"], default="both")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--do-sample", action="store_true", help="Enable sampling; off by default for stable eval")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-samples", type=int, default=None, help="Optional subset for smoke test")
    p.add_argument("--no-resume", action="store_true", help="Do not reuse existing jsonl predictions")
    p.add_argument("--include-time-in-point-error", action="store_true", help="Include t in point distance")
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

    runner = MolmoRunner(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
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
            "lora_path": args.lora_path,
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
        }
    }

    try:
        if args.mode in {"cap", "both"}:
            cap_data = load_json(cap_val_json)
            cap_jsonl = output_dir / "cap_predictions.jsonl"
            cap_rows = run_inference(
                runner,
                cap_data,
                video_root,
                cap_jsonl,
                max_samples=args.max_samples,
                resume=not args.no_resume,
            )
            summary["caption_metrics"] = compute_caption_metrics(cap_rows)

        if args.mode in {"point", "both"}:
            point_data = load_json(point_val_json)
            point_jsonl = output_dir / "point_predictions.jsonl"
            point_rows = run_inference(
                runner,
                point_data,
                video_root,
                point_jsonl,
                max_samples=args.max_samples,
                resume=not args.no_resume,
            )
            summary["point_metrics"] = compute_point_metrics(
                point_rows,
                include_time=args.include_time_in_point_error,
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
