import json

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

import re

import os



# =================設定區域=================

INPUT_FILE = "results_classmate_va.jsonl"  # 你的 JSONL 檔案路徑

MODEL_ID = "google/gemma-3-12b-it"         # 使用 Gemma 3 12B Instruct 版本

# =========================================



# 檢查檔案是否存在

if not os.path.exists(INPUT_FILE):

    print(f"錯誤: 找不到檔案 {INPUT_FILE}，請確認路徑是否正確。")

    exit()



print(f"Loading model: {MODEL_ID}...")



# 載入模型 (使用 4-bit 量化以節省記憶體，適合 RTX 4090)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(

    MODEL_ID,

    device_map="auto",

    torch_dtype=torch.bfloat16,

    quantization_config={"load_in_4bit": True} 

)



def generate_evaluation(pred, gt):

    """

    構造 Prompt 讓 Gemma 3 進行評估 (針對 Walking/Feeding 與 Motor/Pump 聲音)

    """

    prompt = f"""

You are an expert evaluator for video captioning tasks involving shrimp behavior analysis.

Compare the PREDICTION with the GROUND TRUTH based on two specific criteria:



1. Action: 

   - Check if the shrimp is described as "Walking" (or crawling/moving) OR "Feeding".

   - Does the Prediction correctly identify the specific activity mentioned in the GT?

   - Treat "Crawling", "Walking", "Navigating", and "Moving" as SYNONYMS (Target Action).

   - If GT says "Walking" and Pred says "Crawling", label as TP.



2. Sound: 

   - Check for the presence of background machinery noises.

   - Look for terms like "Motor", "Pump", "Mechanical hum", or "Running noise".



For each criteria, label as:

- "TP" (True Positive): Prediction correctly captures the feature mentioned in GT (e.g., GT says 'feeding', Pred says 'feeding'; GT says 'motor hum', Pred says 'mechanical hum').

- "FP" (False Positive): Prediction claims a feature not present in GT (e.g., Pred says 'feeding' but GT only says 'walking'; Pred says 'pump noise' but GT says 'silent').

- "FN" (False Negative): Prediction misses a feature present in GT (e.g., GT mentions 'feeding', Pred only says 'walking'; GT mentions 'hum', Pred says 'silent').



**Ground Truth:** {gt}

**Prediction:** {pred}



Return ONLY a JSON object. Do not explain. Format:

{{

  "Action": "TP/FP/FN",

  "Sound": "TP/FP/FN"

}}

"""

    messages = [

        {"role": "user", "content": prompt}

    ]

    

    # 產生輸入 (Gemma 3 的 tokenizer 會回傳 BatchEncoding 物件)

    inputs = tokenizer.apply_chat_template(

        messages, 

        return_tensors="pt", 

        add_generation_prompt=True

    ).to("cuda")

    

    # 處理 BatchEncoding 與 Tensor 的差異

    if isinstance(inputs, torch.Tensor):

        outputs = model.generate(

            inputs,

            max_new_tokens=100,

            do_sample=False,

            temperature=0.0

        )

        input_length = inputs.shape[1]

    else:

        outputs = model.generate(

            **inputs,

            max_new_tokens=100,

            do_sample=False,

            temperature=0.0

        )

        input_length = inputs["input_ids"].shape[1]

    

    # 只解碼新生成的部份

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response



def parse_json_response(response_text):

    """

    從 LLM 回應中提取 JSON

    """

    try:

        # 嘗試尋找 JSON 區塊

        match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if match:

            return json.loads(match.group(0))

        return json.loads(response_text)

    except:

        return {"Action": "Error", "Sound": "Error"}



# === 主程式開始 ===

results = []

# 更新統計類別，移除 Object，只保留 Action 和 Sound

stats = {"TP": 0, "FP": 0, "FN": 0}



with open(INPUT_FILE, "r", encoding="utf-8") as f:

    lines = f.readlines()



print(f"Starting evaluation on {len(lines)} samples...")



for line in tqdm(lines):

    if not line.strip(): continue

    

    data = json.loads(line)

    pred_desc = data.get("pred_desc", "")

    gt_desc = data.get("gt_desc", "")

    

    # 呼叫 Gemma 3

    raw_response = generate_evaluation(pred_desc, gt_desc)

    eval_result = parse_json_response(raw_response)

    

    # 統計分數 (只針對 Action 和 Sound)

    for key in ["Action", "Sound"]:

        val = eval_result.get(key, "FN") # 若解析失敗預設為 FN

        if "TP" in val: stats["TP"] += 1

        elif "FP" in val: stats["FP"] += 1

        elif "FN" in val: stats["FN"] += 1



    # 保存結果

    data["llm_eval"] = eval_result

    results.append(data)



# === 計算指標 ===

precision = stats["TP"] / (stats["TP"] + stats["FP"] + 1e-9)

recall = stats["TP"] / (stats["TP"] + stats["FN"] + 1e-9)

f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)



print("\n" + "="*30)

print(f"Evaluation Results (Action: Walk/Feed, Sound: Motor/Pump)")

print(f"Total Labels: {stats['TP'] + stats['FP'] + stats['FN']}")

print(f"TP: {stats['TP']}, FP: {stats['FP']}, FN: {stats['FN']}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1_score:.4f}")

print("="*30)



# 寫入新的 JSONL 檔

output_filename = "results_evaluated.jsonl"

with open(output_filename, "w", encoding="utf-8") as f:

    for res in results:

        f.write(json.dumps(res) + "\n")



print(f"詳細評估結果已寫入: {output_filename}")