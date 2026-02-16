import json
import os
import glob
import re

def create_shrimp_dataset_av(input_folder=".", output_filename="shrimp_dataset_audio_and_video.json"):
    # 定義兩個固定的 Human 問句
    questions = [
        # 問句 1 (對應 id _0)
        "<image>Describe the behavior of the shrimp and identify any sounds.\n?",
        
        # 問句 2 (對應 id _1)
        "<image>Describe the shrimp's specific movements.\nIdentify non-speech underwater sounds. Specifically, detect if there are any impulsive noises (clicks, scratches) or continuous background hums (water flow, motor)."
    ]

    dataset = []

    # 搜尋所有結尾是 _gemini.json 的檔案
    search_pattern = os.path.join(input_folder, "*_gemini.json")
    files = glob.glob(search_pattern)
    files.sort()

    print(f"找到 {len(files)} 個檔案，開始處理...")

    for file_path in files:
        filename = os.path.basename(file_path)
        match = re.match(r"(\d+)_gemini\.json", filename)
        
        if not match:
            print(f"跳過格式不符的檔案: {filename}")
            continue
            
        file_id = match.group(1) # 例如 "00558"
        
        # 設定影片與音檔名稱
        video_filename = f"{file_id}.mp4"
        audio_filename = f"{file_id}.wav"  # 加入這行，對應你的 .wav 檔

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 整合 GPT 的回答 (Video + Audio + Fused)
            v_desc = data.get("video_only_desc", "")
            a_desc = data.get("audio_only_desc", "")
            f_desc = data.get("fused_desc", "")
            
            # 使用換行連接
            combined_desc = f"{v_desc}\n{a_desc}\n{f_desc}".strip()

            for i, question in enumerate(questions):
                entry = {
                    "id": f"{file_id}_{i}",
                    
                    # 同時指定 video 和 audio
                    "video": video_filename,
                    "audio": audio_filename,  # 這裡明確指定音檔路徑
                    
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": combined_desc
                        }
                    ]
                }
                dataset.append(entry)
                
        except Exception as e:
            print(f"讀取檔案 {filename} 時發生錯誤: {e}")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"處理完成！已生成 {output_filename}")

if __name__ == "__main__":
    create_shrimp_dataset_av()
