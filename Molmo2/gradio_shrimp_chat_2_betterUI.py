import os
import re
import json
import gc
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


BASE_MODEL_PATH = "../models/molmo2-4b"
# LORA_PATH = "./molmo_runs/shrimp_lora_e1_0519"
LORA_PATH = "./molmo_runs/shrimp_lora_full"
# LORA_PATH = "/media/sdb/dozen/molmo_runs/shrimp_lora_shrimp_lora_masked_e1"
# LORA_PATH = "/media/sdb/dozen/molmo_runs/shrimp_lora_full"
# LORA_PATH = "/media/sdb/dozen/molmo_runs/shrimp_lora_e3"
LOG_PATH = "log0905.jsonl"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 128

processor = None
base_model = None
lora_model = None


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_processor():
    global processor
    if processor is None:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
        )
        print("Processor loaded.")
    return processor


def load_base_model():
    global base_model
    if base_model is None:
        print("Loading BASE model...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            dtype=DTYPE,
            device_map=DEVICE,
        )
        base_model.eval()
        print("BASE model loaded.")
    return base_model


def load_lora_model():
    global lora_model
    if lora_model is None:
        print("Loading LoRA model...")
        lora_base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            dtype=DTYPE,
            device_map=DEVICE,
        )
        lora_model = PeftModel.from_pretrained(lora_base, LORA_PATH)
        lora_model.eval()
        print("LoRA model loaded.")
    return lora_model


def load_all_models_manual():
    """按鈕觸發的提前載入函式"""
    try:
        get_processor()
        load_base_model()
        load_lora_model()
        return "✅ 所有模型已成功載入 GPU，可以開始推論！"
    except Exception as e:
        return f"❌ 載入失敗: {str(e)}"


def normalize_video_input(video_file: Any, local_path: str) -> Optional[str]:
    # 優先使用手動輸入的本機路徑，略過上傳延遲
    if local_path and local_path.strip():
        clean_path = local_path.strip()
        if os.path.exists(clean_path):
            return clean_path
        else:
            print(f"Warning: Local path does not exist: {clean_path}")

    # 退回使用 Gradio 上傳的影片
    if video_file is None:
        return None
    if isinstance(video_file, str) and os.path.exists(video_file):
        return video_file
    if isinstance(video_file, dict):
        for key in ["video", "name", "path"]:
            value = video_file.get(key)
            if isinstance(value, str) and os.path.exists(value):
                return value
    return None


def get_model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def run_model(model, processor, video_path: str, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt_text],
        videos=[video_path],
        return_tensors="pt",
        padding=True,
    )

    print("\n===== TOKEN DEBUG =====")

    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
        else:
            print(k, v)

    print("text token length =", inputs["input_ids"].shape[1])

    if "video_grid_thw" in inputs:
        print("video_grid_thw =", inputs["video_grid_thw"])

    print("========================")

    model_device = get_model_device(model)
    inputs = {
        k: v.to(model_device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    # do_sample=False 保證了 Temperature=0 的 Greedy Decoding 行為
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, input_len:]
    answer = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return answer


def extract_points_and_clean_text(text: str) -> Tuple[list, str]:
    points = []

    # 1. 解析 Base Model 格式: <points coords="0.0 1 102 331 2 480 819...">...</points>
    def replacer_multi(match):
        coords_str = match.group(1)
        vals = coords_str.strip().split()
        try:
            vals = [float(v) for v in vals]
            # 判斷是否符合 Time + (ID, X, Y) 的結構
            if len(vals) > 0 and len(vals) % 3 == 1:
                t = vals[0]
                for i in range(1, len(vals), 3):
                    _id = vals[i]
                    x = vals[i+1]
                    y = vals[i+2]
                    
                    # Molmo2 座標通常為 0~1000 尺度，轉換為 0.0~1.0
                    if x > 100 or y > 100:
                        x, y = x / 1000.0, y / 1000.0
                    elif x > 1 or y > 1:
                        x, y = x / 100.0, y / 100.0
                        
                    points.append((x, y, t))
        except Exception as e:
            print("Error parsing multi-coords:", e)
            
        return "" # 回傳空字串將標籤抹除

    # 2. 解析 LoRA 格式: <point x="0.5" y="0.5" t="0.0"></point>
    def replacer_single(match):
        tag_str = match.group(0)
        x_match = re.search(r'x="([0-9.]+)"', tag_str)
        y_match = re.search(r'y="([0-9.]+)"', tag_str)
        t_match = re.search(r't="([0-9.]+)"', tag_str)
        
        if x_match and y_match:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            t = float(t_match.group(1)) if t_match else 0.0
            
            if x > 100 or y > 100:
                x, y = x / 1000.0, y / 1000.0
            elif x > 1 or y > 1:
                x, y = x / 100.0, y / 100.0
                
            points.append((x, y, t))
        return ""

    # 執行 Regex 替換並抽取
    pattern_multi = r'<points\s+coords="([^"]+)">(.*?)</points>'
    clean_text = re.sub(pattern_multi, replacer_multi, text, flags=re.DOTALL)
    
    pattern_single = r'<point\b[^>]*>(.*?)</point>|<point\b[^>]*/>'
    clean_text = re.sub(pattern_single, replacer_single, clean_text, flags=re.DOTALL)
    
    # 清理多餘的空白換行
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text).strip()
    
    return points, clean_text


def draw_points_on_first_frame(video_path: str, points, prefix: str) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        return None

    h, w = frame.shape[:2]

    for idx, (x, y, _t) in enumerate(points, start=1):
        # 防呆，避免座標超出影像範圍
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        px = int(x * w)
        py = int(y * h)

        cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{prefix}{idx}",
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    out_path = f"/tmp/{prefix}_{time.time_ns()}.jpg"
    cv2.imwrite(out_path, frame)
    return out_path


def save_log(video_path: str, question: str, base_answer: str, lora_answer: str) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "video": video_path,
        "question": question,
        "base": base_answer,
        "lora": lora_answer,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_both(video_file: Any, local_video_path: str, question: str, run_base: bool):
    if not question or not question.strip():
        return "請先輸入問題。", "", None, None, "尚未執行"

    video_path = normalize_video_input(video_file, local_video_path)
    if video_path is None:
        return "請先上傳影片或輸入有效的本機路徑。", "", None, None, "尚未執行"

    proc = get_processor()

    base_answer = ""
    if run_base:
        try:
            base = load_base_model()
            base_answer = run_model(base, proc, video_path, question)
        except Exception as e:
            base_answer = f"[BASE ERROR] {type(e).__name__}: {e}"

    try:
        lora = load_lora_model()
        lora_answer = run_model(lora, proc, video_path, question)
    except Exception as e:
        lora_answer = f"[LORA ERROR] {type(e).__name__}: {e}"

    base_points, clean_base_text = extract_points_and_clean_text(base_answer) if run_base else ([], "")
    lora_points, clean_lora_text = extract_points_and_clean_text(lora_answer)

    base_img = draw_points_on_first_frame(video_path, base_points, "B") if base_points else None
    lora_img = draw_points_on_first_frame(video_path, lora_points, "L") if lora_points else None

    save_log(video_path, question, base_answer, lora_answer)

    status = (
        f"完成 | 影片來源: {os.path.basename(video_path)} | "
        f"BASE 點數: {len(base_points)} | "
        f"LORA 點數: {len(lora_points)}"
    )

    return clean_base_text, clean_lora_text, base_img, lora_img, status


def clear_outputs():
    return "", "", None, None, "已清空"


def unload_models():
    global base_model, lora_model
    if base_model is not None:
        del base_model
        base_model = None
    if lora_model is not None:
        del lora_model
        lora_model = None
    clear_gpu()
    return "模型已卸載，GPU cache 已清理。"


# --- 這裡我將 CSS 選擇器寫得更全面，確保所有種類的元件標籤都被覆蓋 ---
custom_css = """
.gradio-container label span, 
.gradio-container .label-text, 
.gradio-container .built-in-label span {
    font-size: 22px !important; 
    font-weight: bold !important;
}
"""

with gr.Blocks(title="Shrimp Base vs LoRA Demo", theme=gr.themes.Soft(text_size="lg"), css=custom_css) as demo:
    gr.Markdown("# 🦐 Shrimp Video QA / Pointing Demo")
    gr.Markdown("只保證在 `molmo2_cu124` 環境執行。支援本機影片路徑輸入以略過上傳時間。")

    with gr.Row():
        show_base_cb = gr.Checkbox(label="啟用並顯示 Base Model 結果 (預設隱藏，會增加推論時間)", value=False)
        
    with gr.Row():
        with gr.Column(scale=1):
            
            # 隱藏選項A
            local_video_path = gr.Textbox(
                label="選項 A: 影片本機路徑", 
                placeholder="/media/sdb/dozen/.../test.mp4",
                lines=1,
                visible=False
            )
            video = gr.Video(label="上傳影片")
            
            question = gr.Textbox(
                label="輸入問題",
                lines=1,
                max_lines=3,
                placeholder="例如：How many shrimps are there in this video?",
            )

            # # 將按鈕放進摺疊選單中縮小高度
            # with gr.Accordion("範例問題 (點擊展開)", open=False):
            #     with gr.Row():
            #         btn_count = gr.Button("Count shrimp", size="sm")
            #         btn_left = gr.Button("Left shrimp", size="sm")
            #         btn_center = gr.Button("Center shrimp", size="sm")

            #     with gr.Row():
            #         btn_activity = gr.Button("Activity", size="sm")
            #         btn_right = gr.Button("Right-most", size="sm")
            #         btn_visible = gr.Button("Visible shrimp", size="sm")

            # 將控制按鈕合併為一行
            with gr.Row():
                # load_btn = gr.Button("Load Models", variant="secondary")
                # unload_btn = gr.Button("Unload Models", variant="stop")
                run_btn = gr.Button("執行", variant="primary")
                # clear_btn = gr.Button("Clear")

            status = gr.Textbox(label="系統狀態", value="等待操作...", interactive=False)

        # 輸出結果區塊動態調整高度與略微增加寬度佔比
        with gr.Column(scale=1.2, visible=False) as base_col:
            base_out = gr.Textbox(label="BASE 輸出結果", lines=1, max_lines=15, autoscroll=True)
            base_img = gr.Image(label="BASE 標記點")

        with gr.Column(scale=1.2):
            lora_out = gr.Textbox(label="Shrimp MLLM的回答", lines=1, max_lines=15, autoscroll=True)
            lora_img = gr.Image(label="標記點")

    show_base_cb.change(
        fn=lambda is_checked: gr.update(visible=is_checked),
        inputs=[show_base_cb],
        outputs=[base_col]
    )

    # btn_count.click(lambda: "How many shrimps are there in this video?", inputs=[], outputs=question)
    # btn_left.click(lambda: "How many shrimps are closest to the left?", inputs=[], outputs=question)
    # btn_center.click(lambda: "How many shrimps are closest to the center of the screen?", inputs=[], outputs=question)
    # btn_activity.click(lambda: "What is the primary activity observed among the shrimp?", inputs=[], outputs=question)
    # btn_right.click(lambda: "How many shrimps are furthest to the right?", inputs=[], outputs=question)
    # btn_visible.click(lambda: "How many shrimps are visible in the video?", inputs=[], outputs=question)

    # # 綁定提前載入按鈕
    # load_btn.click(
    #     fn=load_all_models_manual,
    #     inputs=[],
    #     outputs=[status]
    # )

    run_btn.click(
        fn=run_both,
        inputs=[video, local_video_path, question, show_base_cb],
        outputs=[base_out, lora_out, base_img, lora_img, status],
    )

    # clear_btn.click(
    #     fn=clear_outputs,
    #     inputs=[],
    #     outputs=[base_out, lora_out, base_img, lora_img, status],
    # )

    # unload_btn.click(
    #     fn=unload_models,
    #     inputs=[],
    #     outputs=status,
    # )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)