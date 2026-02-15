#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import shutil
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# 1. 設定區
# =========================
DATA_ROOT = "影片"                  # 資料來源
OUTPUT_JSON = "shrimp_dataset.json" # 輸出 JSON
OUTPUT_IMAGE_DIR = "images_out"     # 圖片輸出目錄

# [需求 2] 固定 Model ID
MODEL_ID = "google/gemma-3-12b-it"   
DIALOGUE_LANGUAGE = "English"       

MAX_ITERS = 3                       # 每個對話集最大修正次數
GEN_DIALOGUE_MIN = 1                # 每次生成至少幾組獨立對話
GEN_DIALOGUE_MAX = 2                # 每次生成最多幾組獨立對話

# [需求 1] 採樣間隔
# 假設影片 FPS=30，若要每 1 秒取 1 張，設 30
# 若要每 30 秒取 1 張，設 900
SAMPLE_INTERVAL = 150  # [修正] 這裡設定為每 150 張取 1 張

# =========================
# 2. 讀取 State (修正版：支援多物件)
# =========================
def get_shrimp_state(json_path: str) -> str:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = 0
        swim = 0
        feed = 0
        swim_feed = 0
        unknown = 0

        for shape in data.get("shapes", []):
            attributes = shape.get("attributes", {})
            if "state" not in attributes:
                continue

            total += 1
            states = attributes["state"]
            if not isinstance(states, list):
                states = [states]
            states = [str(x).strip().lower() for x in states]

            sset = set(states)
            if "swimming" in sset and "feeding" in sset:
                swim_feed += 1
            elif "swimming" in sset:
                swim += 1
            elif "feeding" in sset:
                feed += 1
            else:
                unknown += 1

        if total == 0:
            return "unknown"

        return f"total={total}; swimming={swim}; feeding={feed}; swimming+feeding={swim_feed}; other={unknown}"

    except Exception as e:
        print(f"[WARN] Error reading {json_path}: {e}")
        return "unknown"


# =========================
# 3. LLM 初始化
# =========================
def load_model_and_tokenizer(model_id: str):
    print(f"[INFO] Loading model: {model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer

MODEL, TOKENIZER = load_model_and_tokenizer(MODEL_ID)

def call_llm(prompt: str, max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]

    inputs = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(MODEL.device)
        model_inputs = {"input_ids": input_ids}
    else:
        model_inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=TOKENIZER.eos_token_id,
    )

    if temperature and temperature > 0:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        ))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out = MODEL.generate(**model_inputs, **gen_kwargs)

    prompt_len = model_inputs["input_ids"].shape[-1]
    gen_ids = out[0][prompt_len:]
    return TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

def extract_json_array(text: str) -> Optional[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start: return None
    try: return json.loads(text[start:end+1].strip())
    except: return None

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start: return None
    try: return json.loads(text[start:end+1].strip())
    except: return None

# =========================
# 4. Agent Prompts
# =========================
def build_generate_prompt(state: str, extra_notice: str) -> str:
    return f"""
You are an assistant that must produce ONLY valid JSON.
You are given a summary of shrimp counts by activity in the image. Use this summary to write natural {DIALOGUE_LANGUAGE} dialogues about the **overall scene and behaviors**.

Output format:
- A single JSON array of dialogue arrays: [ [{{...}}, {{...}}], [{{...}}, {{...}}] ]
- Each dialogue object: {{ "from": "human"|"gpt", "value": "text" }}

Rules:
1. Generate {GEN_DIALOGUE_MIN}-{GEN_DIALOGUE_MAX} INDEPENDENT dialogues.
2. Each dialogue: 1-5 turns.
3. Every dialogue MUST start with a human message whose value begins exactly with "<image>\\n".
4. Speakers must alternate strictly: human, gpt, human, gpt...
5. Content: 
   - Since there are multiple shrimps, describe the **variety** of activities.
   - Use phrases like "Some shrimps are...", "One appears to be...", "Most of them are...".
   - Base your descriptions only on the provided activity summary (counts).
6. RESTRICTION: Do NOT mention specific IDs like "shrimp_1" or "shrimp_5". Just refer to them as "a shrimp", "another one", or "the group".
7. RESTRICTION: Do NOT mention "tags", "labels", "metadata".
8. OBSERVATION: Phrase answers conservatively: "appears/seems/likely".
9. Activity Summary (counts): {state}

Background Knowledge:
- **Swimming**: Uses pleopods (swimmerets), body extended, tail fan spread.
- **Feeding**: Uses legs to pick particles from bottom, head tilted down, stationary or slow moving.

Example Q&A (Do not copy verbatim):
- Q: "<image>\\nWhat are the shrimps doing in this image?"
  A: "There are multiple shrimps in the scene. Most appear to be swimming, while at least one seems to be feeding near the bottom."
- Q: "<image>\\nIs there any feeding behavior?"
  A: "Yes, I can see a shrimp that appears to be foraging on the substrate, distinct from the others that are swimming."

Additional constraints:
{extra_notice}

Return ONLY JSON.
""".strip()

def build_check_prompt(conversations_json_text: str) -> str:
    return f"""
You are given a JSON array of dialogues about shrimp activity and condition.

Your task is to determine whether the dialogues are acceptable (PASS) or need refinement (FAIL).

#### dialogues
{conversations_json_text}

#### task_note (you MUST follow all rules strictly)

The entire set is acceptable (PASS) only if ALL answers:

1. Are grounded in plausible visual observation 
   (e.g., posture, body orientation, limb movement, relative position in frame).
2. Do NOT mention or rely on non-visual metadata such as:
   "tags", "labels", "annotations", "dataset", "provided list", or similar.
3. Do NOT mention specific shrimp IDs such as "shrimp_1", "shrimp_2", etc.
4. Do NOT claim medical diagnosis, lab confirmation, or external knowledge.
5. If describing activity (e.g., swimming, feeding), 
   use conservative phrasing such as:
   - "appears to be"
   - "seems to be"
   - "likely"
6. Do NOT introduce details that cannot reasonably be inferred from a still image.
7. Strictly follow dialogue structure:
   - First message must be from "human"
   - It must begin exactly with "<image>\n"
   - Speakers must alternate human/gpt
   - 1–5 turns per dialogue
   - Only keys allowed: "from", "value"

If ANY rule is violated, mark the whole set as FAIL.

#### Output format

Return ONLY this JSON:

{{
  "reason": "short justification",
  "result": "PASS" | "FAIL"
}}

No additional text.

""".strip()

def build_refine_prompt(last_gen_text: str, fail_reason: str) -> str:
    return f"""
You are given dialogues that failed validation and the reason for failure.

Fail reason:
{fail_reason}

Failed generation (snippet):
{last_gen_text[:800]}


Your task:
1. Identify the main structural or content problems.
2. Write a short, structured NOTICE block that will be appended to the next generation prompt.
3. The notice must:
   - Enforce JSON-only output
   - Enforce "<image>\n" at the start
   - Enforce strict alternation of speakers
   - Prohibit mention of tags/metadata
   - Require conservative activity phrasing ("appears", "seems")

Return ONLY the NOTICE text.
No JSON.
No explanations.
""".strip()


def parse_counts(state: str) -> Dict[str, int]:
    out = {}
    try:
        parts = [p.strip() for p in state.split(";")]
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                out[k.strip()] = int(v.strip())
    except Exception:
        pass
    return out


# =========================
# 5. Agent Loop
# =========================
def generate_dialogues_with_agents(state: str) -> List[List[Dict[str, str]]]:
    """
    回傳 List[List[Dict]]，也就是多組對話的列表。
    """
    extra_notice = ""
    last_gen_text = ""

    for attempt in range(1, MAX_ITERS + 1):
        print(f"   -> Agent Attempt {attempt}...")
        
        # 1. Generator
        gen_prompt = build_generate_prompt(state, extra_notice)
        gen_text = call_llm(gen_prompt, max_new_tokens=1024, temperature=0.7)
        last_gen_text = gen_text
        
        dialogues = extract_json_array(gen_text)
        if not dialogues or not isinstance(dialogues, list):
            fail_reason = "Output is not a valid JSON array."
            print(f"      [FAIL] {fail_reason}")
            # Refine
            refine_prompt = build_refine_prompt(last_gen_text, fail_reason)
            extra_notice = call_llm(refine_prompt, max_new_tokens=150, temperature=0.2)
            continue

        # 2. Checker
        check_prompt = build_check_prompt(json.dumps(dialogues)) # 確保傳入的是乾淨的 JSON string
        check_text = call_llm(check_prompt, max_new_tokens=150, temperature=0.0)
        check_obj = extract_json_object(check_text)

        if check_obj and check_obj.get("result") == "PASS":
            print("      [PASS] Quality check passed.")
            return dialogues
        
        fail_reason = check_obj.get("reason", "Unknown check failure") if check_obj else "Checker output invalid"
        print(f"      [FAIL] {fail_reason}")

        # 3. Refiner
        refine_prompt = build_refine_prompt(last_gen_text, fail_reason)
        extra_notice = call_llm(refine_prompt, max_new_tokens=150, temperature=0.2)

    # Fallback if all fail
    print("   -> [WARN] All attempts failed. Using fallback.")

    # 1) unknown 直接回傳
    if state == "unknown":
        ans = "The scene appears to contain multiple shrimps, but their specific activity is unclear from this frame."
        return [[
            {"from": "human", "value": "<image>\nWhat are the shrimps doing in this image?"},
            {"from": "gpt", "value": ans}
        ]]

    # 2) 解析 counts，決定回答內容
    counts = parse_counts(state)
    sw = counts.get("swimming", 0)
    fd = counts.get("feeding", 0)
    sf = counts.get("swimming+feeding", 0)

    if (fd + sf) > 0 and (sw + sf) > 0:
        ans = "The scene appears to contain multiple shrimps showing mixed behaviors. Some likely are swimming, and a few may be feeding or foraging near the bottom."
    elif (fd + sf) > 0:
        ans = "The shrimps appear to be mostly feeding or foraging near the bottom, using their legs to pick at particles."
    elif (sw + sf) > 0:
        ans = "The shrimps appear to be mostly swimming, likely propelling themselves with their swimmerets."
    else:
        ans = "The scene appears to contain multiple shrimps, but their specific activity is unclear from this frame."

    return [[
        {"from": "human", "value": "<image>\nWhat are the shrimps doing in this image?"},
        {"from": "gpt", "value": ans}
    ]]

# =========================
# 6. 主程式
# =========================
def main():
    # [修正] 每次執行前先清空輸出目錄，避免混到舊圖
    if os.path.exists(OUTPUT_IMAGE_DIR):
        shutil.rmtree(OUTPUT_IMAGE_DIR)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    dataset = []
    global_counter = 1 

    video_folders = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    print(f"[INFO] Found video folders: {video_folders}")

    for video_id in video_folders:
        video_path = os.path.join(DATA_ROOT, video_id)
        image_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))

        print(f"Processing folder {video_id}: Total {len(image_files)} images.")

        for i, img_path in enumerate(image_files):
            # [修正 1] 採樣邏輯：每 SAMPLE_INTERVAL 張取 1 張
            if i % SAMPLE_INTERVAL != 0:
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(video_path, f"{base_name}.json")

            state = "unknown" # 預設值
            if os.path.exists(json_path):
                s = get_shrimp_state(json_path)
                if s: state = s

            # 複製並重命名圖片 (例如 00001.jpg)
            # 使用全域編號，確保檔名唯一
            new_img_name = f"{global_counter:05d}.jpg"
            shutil.copy2(img_path, os.path.join(OUTPUT_IMAGE_DIR, new_img_name))

            # 生成多組對話 (List of Lists)
            dialogues_list = generate_dialogues_with_agents(state)

            # [修正 2] Data Augmentation 邏輯
            # 將 LLM 生成的 1-2 組獨立對話，拆成獨立的資料
            for j, single_dialogue in enumerate(dialogues_list):
                entry = {
                    "id": f"{global_counter:05d}_{j}", # ID 必須唯一
                    "image": new_img_name,             # 圖片檔名相同
                    "conversations": single_dialogue
                }
                dataset.append(entry)

            print(f"   -> [Saved] {new_img_name} (Origin: {base_name}) | State: {state} | Generated {len(dialogues_list)} variations")
            global_counter += 1

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Total dataset entries: {len(dataset)}")
    print(f"Images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"JSON saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()