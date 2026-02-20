import json
import os
import subprocess
import re

# ========= 基本設定 =========
MODEL_NAME = "gemma3:12b"
DATA_PATH = "/home/yita/Desktop/tmp/12_28/影片"
VIDEO_NAMES = [f"{i:04d}" for i in range(37, 42)]            # 對應 0001.mp4 ->0038.mp4
FRAME_INTERVAL = 60                       # 每 幾 frame 取一個,這邊是2s取一frame,一次16frame資料
OUTPUT_DIR = "./VQA.json"
MAX_RETRIES = 3  

# ========= Ollama 呼叫函式 =========
def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()
#重心計算
def get_shrimp_center(polygon):
    """計算精確面積重心，確保座標在蝦子身上"""
    n = len(polygon)

    # --- 加上防呆機制 ---
    if n == 0: 
        return [0.0, 0.0]

    if n < 3: return [round(sum(p[0] for p in polygon)/n, 3), round(sum(p[1] for p in polygon)/n, 3)]
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        cp = x0 * y1 - x1 * y0
        area += cp
        cx += (x0 + x1) * cp
        cy += (y0 + y1) * cp
    area *= 0.5
    if abs(area) < 1e-9: return [round(polygon[0][0], 3), round(polygon[0][1], 3)]
    return [round(cx / (6.0 * area), 3), round(cy / (6.0 * area), 3)]
def calculate_movement(frames, shrimp_id):
    """輸入整理好的 frames list，輸出每隻蝦的移動描述"""
    positions = [s['pos'] for f in frames for s in f['shapes'] if s['id'] == shrimp_id]
    if not positions:
        return None
    
    start = positions[0]
    end = positions[-1]

    # 判斷是否明顯移動
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    if dx < 0.01 and dy < 0.01:
        return "stationary"
    else:
        return f"moving from [{round(start[0],3)}, {round(start[1],3)}] to [{round(end[0],3)}, {round(end[1],3)}]"
def clean_output_json(text: str) -> str:
    """
          移除 ```json 開頭與 ``` 結尾，只留下純 JSON 文字
    """
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)
    return text.strip()
def generate_summary(json_folder, video_name):
    """
    讀取 JSON frames，計算 shrimp_summary 和 cleaned_frames_data
    回傳:
        shrimp_summary: dict, 每隻蝦子的狀態列表
        cleaned_frames_data: list, 精簡 frame 資料 (位置 + 狀態)
        video_metadata: dict, 影片寬高與總 frame 數
        gen_header, generate agent用的
    """
    # 只讀 .json，排序保證時間順序
    frame_files = sorted(f for f in os.listdir(json_folder) if f.endswith(".json"))

    # --- 修改點 1: 在這裡進行資料瘦身 ---
    # 在迴圈外先拿第一幀的寬高 (假設影片解析度不變)
    with open(os.path.join(json_folder, frame_files[0]), "r") as f:
        first_frame = json.load(f)
        video_metadata = {
            "width": first_frame.get("imageWidth"),
            "height": first_frame.get("imageHeight"),
            "total_frames": len(frame_files)
        }

    cleaned_frames_data = []
    shrimp_summary = {}

    for idx, fname in enumerate(frame_files):
        if idx % FRAME_INTERVAL != 0:
            continue
            
        with open(os.path.join(json_folder, fname), "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            current_frame = {"f": fname, "shapes": []}
            
            for s in raw_data.get('shapes', []):
                label = s.get('label')
                state = s.get('attributes', {}).get('state', [])
                
                # if label not in shrimp_summary:
                #     shrimp_summary[label] = state
                if label not in shrimp_summary:
                    shrimp_summary[label] = set(state)
                else:
                    shrimp_summary[label].update(state)

                abs_pos = get_shrimp_center(s.get('points', []))
                rel_pos = [
                    round(abs_pos[0] / video_metadata['width'], 3),
                    round(abs_pos[1] / video_metadata['height'], 3)
                ]

                current_frame["shapes"].append({
                    "id": label,
                    "pos": rel_pos,  # 存入相對座標 [x_rel, y_rel]
                    "state": state
                })
            cleaned_frames_data.append(current_frame)

    # 1. 集合運算
    swim_ids = set([i for i, s in shrimp_summary.items() if "swimming" in s])
    feed_ids = set([i for i, s in shrimp_summary.items() if "feeding" in s])
    active_ids = swim_ids | feed_ids
    
    # 2. 取得各項精確數值
    total_val = len(shrimp_summary)
    resting_val = total_val - len(active_ids)
    overlap_val = len(swim_ids & feed_ids)

    # 3. 建立每個狀態對應的座標列表
    state_coords = {
        "resting": [],
        "swimming": [],
        "feeding": [],
        "swim+feed": []
    }

    for shrimp_id, states in shrimp_summary.items():
        # --- 補上這兩行：若狀態為空，視為 resting ---
        if not states:
            shrimp_summary[shrimp_id].add("resting")

        first_frame_pos = None
        # 取第一個出現的 frame 位置
        for f in cleaned_frames_data:
            for s in f['shapes']:
                if s['id'] == shrimp_id:
                    first_frame_pos = s['pos']
                    break
            if first_frame_pos is not None:
                break
        
        if "swimming" in states and "feeding" in states:
            state_coords["swim+feed"].append({"id": shrimp_id, "pos": first_frame_pos})
        elif "swimming" in states:
            state_coords["swimming"].append({"id": shrimp_id, "pos": first_frame_pos})
        elif "feeding" in states:
            state_coords["feeding"].append({"id": shrimp_id, "pos": first_frame_pos})
        else:
            state_coords["resting"].append({"id": shrimp_id, "pos": first_frame_pos})

    # --- 重建 calc_text（每個狀態都帶座標） ---
    parts = []
    total_val = len(shrimp_summary)
    parts.append(f"There are {total_val} shrimp in total.")
    
    # 逐狀態列舉
    state_texts = []
    for state_name, shrimp_list in state_coords.items():
        if shrimp_list:
            coords_str = ", ".join([f"[{round(s['pos'][0],3)}, {round(s['pos'][1],3)}]" for s in shrimp_list])
            if state_name == "swim+feed":
                state_texts.append(f"{len(shrimp_list)} are both swimming and feeding at coordinates {coords_str}")
            else:
                state_texts.append(f"{len(shrimp_list)} are {state_name} at coordinates {coords_str}")
    if state_texts:
        parts.append(". ".join(state_texts) + ".")

    calc_text = " ".join(parts)

    json_str = json.dumps({"metadata": video_metadata, "data": cleaned_frames_data}, ensure_ascii=False)
    video_path = f"12_28/{video_name}.mp4"

    # ========= Prompt =========
    
    # =========  Step 1: 計算每隻蝦的移動並保存座標 =========
    shrimp_movement = {}
    for shrimp_id in shrimp_summary.keys():
        movement_desc = calculate_movement(cleaned_frames_data, shrimp_id)
        # 保存每隻蝦的移動資訊
        if movement_desc != "stationary":
            shrimp_movement[shrimp_id] = movement_desc
        else:
            # stationary 也存，方便生成數量問題時帶座標（start = end）
            # 直接取第一個位置
            positions = [s['pos'] for f in cleaned_frames_data for s in f['shapes'] if s['id'] == shrimp_id]
            if positions:
                pos = positions[0]
                shrimp_movement[shrimp_id] = f"stationary at [{round(pos[0],3)}, {round(pos[1],3)}]"

    # --- Step 2: 建立 movement 說明文字 ---
    # movement_lines = [f"{k}: {v}" for k, v in shrimp_movement.items()]
    # 改成這樣（隱藏 ID，直接說有一隻蝦子...）
    movement_lines = [f"A shrimp is {v}" for k, v in shrimp_movement.items()]
    movement_text = "Shrimp Movement Info (for all shrimp, including stationary):\n" + "\n".join(movement_lines) + "\n\n"


    # --- Step 3: 組合 prompt header ---
    gen_header = (
        f"Video Path: {video_path}\n"
        f"Observed Summary: {calc_text}\n"
        f"{movement_text}"
        f"Tracking Data: {json_str}\n\n"
    )
    # ========= Step 4: 輸出 shrimp_summary.json =========
    shrimp_summary_output = {}
    for shrimp_id in shrimp_summary.keys():
        # 取第一個出現的 frame
        start_pos = None
        end_pos = None
        for f in cleaned_frames_data:
            for s in f['shapes']:
                if s['id'] == shrimp_id:
                    if start_pos is None:
                        start_pos = s['pos']
                    end_pos = s['pos']
        shrimp_summary_output[shrimp_id] = {
            "start": start_pos,
            "end": end_pos,
            "state": list(shrimp_summary[shrimp_id])#modified # shrimp_summary[shrimp_id]
        }

    
    return shrimp_summary_output, gen_header
def generate_dialogues(video_name, header, suggestion_text=""):
    """
    根據 header 生成對話
    suggestion_text: 可選，額外提供 refine 建議給模型
    回傳:
        dialogues_json_str: str, 生成的 JSON 對話文字 (純文字)
    """
    template = """
You are an assistant that knows only the information provided in the annotations of multiple frames from a single video. If there is a Suggestion section, you MUST follow all instructions in it exactly. Do not ignore or skip any details. All dialogues must reflect the suggestions precisely. Refer to these tags, generate 10 independent 3-5 turn question-and-answer dialogues about the shrimp's position, quantity, and activity (state). Each human question should refer to information provided in the previous assistant's answer whenever possible. 
Ensure the conversation flows naturally from one turn to the next. Follow exactly:

1. **Language**: Use English.

2. **Format**: A single JSON array of dialogue objects. Each object must contain:
    - `"id"`: unique string or number for this dialogue
    - `"video"`: the video path string
    - `"conversations"`: an array of objects with `"from"` and `"value"` alternating human and gpt

3. **Structure**:
    - Generate 3-5 turn dialogues per video.
    - Questions must be about shrimp position, quantity, activity, or movement.
    - Do not reference or hint at tags, frames, or metadata.
    - Use varied question lengths (≤10 words, 11-20 words, >20 words) and matching answer detail.
    - Do not invent numbers; use **Observed Summary** exactly.

4. **Content Rules**:

    - **Authority**: Always follow the Observed Summary for quantities and states. Never recalc.
    
    - **Positions and Coordinates**:
        - All answers about quantity or state MUST include coordinates of each shrimp.
        - Coordinates must match the ones in the Observed Summary exactly.
        - When describing a specific shrimp, always include its coordinate.
        - Natural language descriptions may be used only when no coordinates are asked.

    - **States**:
        - List all observed states per shrimp.
        - **OVERLAPPING STATES RULE**: Only describe overlapping states (e.g., swimming + feeding) if they exist. Do not split overlapping states into separate single-state counts. Only mention single states if pure single-state shrimp exist.
        - Zero-count states: Omit in general summary; answer explicitly "none" if asked about a 0-count state.

    - **Movement**:
        - Describe movement using the exact "Shrimp Movement Info" provided.
        - Include start and end coordinates for each moving shrimp.
        - If a shrimp is stationary, mention it with its coordinate: "stationary at [x, y]".

    - **Yes/No Questions**:
        - Start with "Yes" or "No".
        - Specify exact number of shrimp and their coordinates.
        - Be concise and natural; do not mention unrelated states.

    - **Question Diversity**:
        - Include short (≤10 words), medium (11–20 words), and long (>20 words) questions in each set.
        - Vary wording and phrasing; do not repeat previous questions verbatim.

5. **Examples of correct answers**:
    - "There are five shrimp observed at coordinates [0.123,0.456], [0.234,0.567], [0.847,0.652], [0.291,0.745], and [0.200,0.380]. One shrimp is resting, and four are both swimming and feeding."
    - "The shrimp at [0.847,0.652] is swimming near the upper-right area."
    - "Yes, the two shrimp at [0.312,0.423] and [0.421,0.512] are resting near the center."
    - "The shrimp at [0.512,0.423] is resting at the top-left corner."
    - "The shrimp at [0.623,0.314] is feeding in the center of the video."
    - "The shrimp at [0.512,0.423] is both swimming and feeding."
    - "The shrimp at [0.847,0.652] is moving from [0.847,0.652] to [0.597,0.819]."
    - "The two shrimp at [0.234,0.345] and [0.412,0.567] are moving in opposite directions."
    - "The shrimp at [0.732,0.618] is positioned close to the substrate and appears to interact with it."

6. **Forbidden**:
    - Do not mention "tags", "labels", metadata, frames, or frame numbers.
    - Do not invent numbers or coordinates.
    - Avoid vague terms like "some" or "several".
    - Do not describe overlapping states individually as single states unless pure single-state shrimp exist.
    - NEVER use internal tracking IDs, numbers, or names (e.g., "shrimp 1", "shrimp number 2", "shrimp_4"). Refer to a specific shrimp strictly by its coordinate (e.g., "The shrimp at [0.123, 0.456]").

7. **Output**:
    - JSON array of 5 dialogue objects, each with "id", "video", "conversations" as described.
"""

    # 4. 組合 Prompt
    prompt = header + template
    if suggestion_text:
        prompt += "\n#### Suggestion:\n" + suggestion_text
    print(f"[處理中] {video_name} ...")
    output = call_ollama(prompt)
    return clean_output_json(output)
def check_dialogues(video_name, dialogues_json, summary_json):
    """
    將對話與 summary 給模型檢查
    回傳:
        check_result: dict, {"reason": "...", "result": "PASS"/"FAIL"}
    """
    # --- prompt 組合給 Gemma3 判斷 ---
    prompt = f"""
### Dialogue Quality Check with Summary

You are given a summary of shrimp positions, movements, and states, along with a JSON array of dialogues.

#### Summary
{summary_json}

#### Dialogues
{dialogues_json}

#### Task:
Check if the dialogues **fully comply with the summary**. A dialogue passes (**PASS**) only if:

1. All shrimp mentioned in the summary are correctly described with their coordinates and states.
2. Movements match the summary information.
3. No external tags, tracking IDs (e.g., "shrimp 1", "shrimp_4"), frame numbers, metadata, or invented information are mentioned.
4. Yes/No questions specify exact numbers and coordinates if relevant.

If any violation occurs, mark **FAIL**.

#### Output format:
Return exactly a JSON object:

{{
  "reason": "short explanation of issues or PASS",
  "result": "PASS" | "FAIL"
}}
"""

    print(f"[檢查 QA] {video_name} ...")
    raw_output = call_ollama(prompt)
    # --- 抓出純 JSON ---
    check_result = clean_output_json(raw_output)
    if not check_result:
        print(f"[錯誤] 無法從模型輸出抓到 JSON: {video_name}")
    return json.loads(check_result)
def generate_refine_suggestion(video_name, dialogues_json, check_data):
    """
    根據 check_result 生成下一次模型使用的 refine suggestion
    回傳:
        refine_suggestion_text: str
    """
    # ======== Prompt 加入 reason ========
    prompt = f"""
### Dialogue-Quality Check and Refine Prompt

You are given a JSON array of dialogues about shrimp position, quantity, and activity (state) for the video "{video_name}" and the reasons why they need refinement.
Your task is to summarize the key issues in these dialogues and give a prompt to prevent the next generation from mentioning or hinting at tags, labels, metadata, or any other information that is not observable in the image.

#### Input dialogues:
{dialogues_json}

#### QA check result:
{json.dumps(check_data, ensure_ascii=False)}

#### Output:
Return a message listing key points and recommendations for generating new dialogues.
"""
    print(f"[生成修正建議] {video_name} ...")
    raw_output = call_ollama(prompt)
    # 把 ```json 或多餘文字清掉
    refine_suggestion = clean_output_json(raw_output)
    return refine_suggestion
# ========= 主流程 =========
if __name__ == "__main__":
    all_dialogues = {}
    for idx, video_name in enumerate(VIDEO_NAMES, start=1):
        json_folder = os.path.join(DATA_PATH, video_name)

        # 1. 生成 summary
        shrimp_summary, gen_header = generate_summary(json_folder, video_name)

        # 初始化 refine 建議
        suggestion_text = ""
        dialogue_passed = False
        retry_count = 0
        while not dialogue_passed and retry_count < MAX_RETRIES:
            # 2. 生成 dialogues (可帶 suggestion)
            dialogues_json_str = generate_dialogues(video_name, gen_header, suggestion_text)

            # 3. 檢查 dialogues
            check_result = check_dialogues(video_name, dialogues_json_str, shrimp_summary)

            if check_result.get("result") == "PASS":
                dialogue_passed = True
            else:
                # 4. 生成 refine suggestion
                suggestion_text = generate_refine_suggestion(video_name, dialogues_json_str, check_result)
                # 下一輪生成 dialogue 時會帶上新的 refine 建議
            retry_count += 1
        if not dialogue_passed:
            print(f"[警告] {video_name} 未通過 QA，使用最後生成的對話")
        # --- 保存對話文字 ---
        all_dialogues[video_name] = json.loads(dialogues_json_str)

        # 每五部影片就先寫一次檔
        if idx % 5 == 0:
            with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
                json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
            print(f"[暫存] 已寫 {idx} 部影片對話到 {OUTPUT_DIR}")

    # 最後再確保完整檔案寫入
    with open(OUTPUT_DIR, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)
    print(f"[完成] 全部 {len(VIDEO_NAMES)} 部影片對話已存入 {OUTPUT_DIR}")

