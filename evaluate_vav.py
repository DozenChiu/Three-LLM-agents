# import os
# import json
# import argparse
# import torch
# from pathlib import Path
# from tqdm import tqdm
# from dataclasses import dataclass
# from typing import Optional

# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer

# from videollava.model.builder import load_pretrained_model
# from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from videollava.conversation import conv_templates
# from videollava.mm_utils import tokenizer_image_token

# # 確保 nltk 資源已下載
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# @dataclass
# class ModelArgs:
#     image_tower: Optional[str] = None
#     video_tower: Optional[str] = None
#     audio_tower: Optional[str] = None
#     mm_vision_select_layer: int = -2
#     mm_vision_select_feature: str = "patch"
#     pretrain_mm_mlp_adapter: Optional[str] = None
#     mm_projector_type: str = "linear"

# def parse_args():
#     parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA model on Shrimp Dataset")
#     parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model or LoRA weights")
#     parser.add_argument("--model-base", type=str, default=None, help="Base model path if using LoRA")
#     parser.add_argument("--eval-json", type=str, required=True, help="Path to the validation json file")
#     parser.add_argument("--av-root", type=str, default="", help="Root directory for audio/video files")
#     parser.add_argument("--conv-mode", type=str, default="llava_v1")
#     parser.add_argument("--max-frames", type=int, default=8)
#     parser.add_argument("--max-new-tokens", type=int, default=256)
#     parser.add_argument("--temperature", type=float, default=0.2)
#     return parser.parse_args()

# def extract_qa(conversations):
#     """從 conversations 中提取 human 的問題與 gpt 的標準答案"""
#     question, answer = "", ""
#     for msg in conversations:
#         if msg["from"] in ["human", "user"] and not question:
#             question = msg["value"].replace("<image>", "").replace("<video>", "").replace("<audio>", "").strip()
#             if "\n" in question:
#                 question = question.split("\n", 1)[1].strip()
#         elif msg["from"] in ["gpt", "assistant"] and not answer:
#             answer = msg["value"].strip()
            
#     return question, answer

# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. 載入模型與處理器
#     print("[INFO] Loading model...")
#     tokenizer, model, processor, _ = load_pretrained_model(
#         model_path=args.model_path,
#         model_base=args.model_base,
#         model_name=args.model_path,
#         load_8bit=False,
#         load_4bit=False
#     )
    
#     # 強制掛載並初始化 Audio Tower
#     audio_tower_name = getattr(model.config, "mm_audio_tower", "LanguageBind/LanguageBind_Audio")
#     image_tower_name = getattr(model.config, "mm_image_tower", "LanguageBind/LanguageBind_Image")
#     video_tower_name = getattr(model.config, "mm_video_tower", "LanguageBind/LanguageBind_Video_merge")
    
#     model_args = ModelArgs(
#         image_tower=image_tower_name,
#         video_tower=video_tower_name,
#         audio_tower=audio_tower_name,
#     )
#     model.model.initialize_vision_modules(model_args)
    
#     # 確保新掛載的模組移至正確的硬體與精度
#     model.to(device=device, dtype=torch.float16)

#     model.eval()
    
#     vt = model.get_video_tower()
#     at = model.get_audio_tower()
#     if not vt.is_loaded:
#         vt.load_model()
#     if not at.is_loaded:
#         at.load_model()
        
#     video_processor = vt.video_processor
#     audio_processor = at.audio_processor
    
#     # 初始化 Scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     smoothie = SmoothingFunction().method4
    
#     results = []
#     bleu_scores = {"1": [], "2": [], "3": [], "4": []}
#     rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
#     # 2. 讀取驗證集資料
#     with open(args.eval_json, "r", encoding="utf-8") as f:
#         eval_data = json.load(f)
        
#     print(f"[INFO] Total evaluation samples: {len(eval_data)}")
    
#     # 3. 推論與計算指標
#     for item in tqdm(eval_data, desc="Evaluating"):
#         video_path = os.path.join(args.av_root, item["video"]) if args.av_root else item["video"]
#         audio_path = os.path.join(args.av_root, item["audio"]) if args.av_root else item["audio"]
        
#         question, ground_truth = extract_qa(item["conversations"])
        
#         # 處理影像與音訊張量
#         try:
#             video_tensor = video_processor(video_path, return_tensors="pt")["pixel_values"][0]
#             # 確保形狀為 [3, T, H, W]
#             if video_tensor.shape[0] != 3 and video_tensor.shape[1] == 3:
#                 video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()
            
#             # 限制/填充幀數
#             T = video_tensor.shape[1]
#             if T > args.max_frames:
#                 video_tensor = video_tensor[:, :args.max_frames, :, :]
#             elif T < args.max_frames:
#                 pad = video_tensor[:, -1:, :, :].repeat(1, args.max_frames - T, 1, 1)
#                 video_tensor = torch.cat([video_tensor, pad], dim=1)
                
#             video_tensor = video_tensor.to(device, dtype=torch.float16)
            
#             audio_tensor = audio_processor(audio_path, return_tensors="pt")
#             if isinstance(audio_tensor, dict):
#                 audio_tensor = list(audio_tensor.values())[0]
#             audio_tensor = audio_tensor.to(device, dtype=torch.float16)
            
#         except Exception as e:
#             print(f"[WARN] Error loading media for {item['id']}: {e}")
#             continue

#         # 準備 Prompt
#         conv = conv_templates[args.conv_mode].copy()
#         special_tokens = [DEFAULT_IMAGE_TOKEN] * args.max_frames
#         if getattr(model.config, "mm_use_im_start_end", False):
#             mm_prefix = "".join([DEFAULT_IM_START_TOKEN + t + DEFAULT_IM_END_TOKEN for t in special_tokens])
#         else:
#             mm_prefix = "".join(special_tokens)
            
#         user_msg = mm_prefix + "\n" + question
#         conv.append_message(conv.roles[0], user_msg)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()
        
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        
#         # 執行生成
#         with torch.inference_mode():
#             # 動態設定生成參數，避免出現 temperature 與 do_sample 衝突的警告
#             gen_kwargs = {
#                 "max_new_tokens": args.max_new_tokens,
#                 "use_cache": True
#             }
#             if args.temperature > 0:
#                 gen_kwargs["temperature"] = args.temperature
#                 gen_kwargs["do_sample"] = True
#             else:
#                 gen_kwargs["do_sample"] = False
                
#             output_ids = model.generate(
#                 input_ids=input_ids,
#                 images={"vision": [video_tensor], "audio": audio_tensor}, # ✅ 關鍵修正：拿掉 .unsqueeze(0)
#                 **gen_kwargs
#             )
            
#         # 處理輸出文字
#         prediction = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        
#         # 4. 計算 Metrics
#         ref_tokens = nltk.word_tokenize(ground_truth.lower())
#         pred_tokens = nltk.word_tokenize(prediction.lower())
        
#         bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
#         bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
#         bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
#         bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
#         bleu_scores["1"].append(bleu1)
#         bleu_scores["2"].append(bleu2)
#         bleu_scores["3"].append(bleu3)
#         bleu_scores["4"].append(bleu4)
        
#         rouge_res = scorer.score(ground_truth, prediction)
#         rouge_scores["rouge1"].append(rouge_res["rouge1"].fmeasure)
#         rouge_scores["rouge2"].append(rouge_res["rouge2"].fmeasure)
#         rouge_scores["rougeL"].append(rouge_res["rougeL"].fmeasure)
        
#         results.append({
#             "id": item["id"],
#             "ground_truth": ground_truth,
#             "prediction": prediction,
#             "metrics": {
#                 "bleu4": bleu4,
#                 "rougeL_f1": rouge_res["rougeL"].fmeasure
#             }
#         })

#     # 5. 輸出總平均
#     print("\n" + "="*40)
#     print("📋 Evaluation Results (Average)")
#     print("="*40)
#     print(f"Total Samples : {len(results)}")
#     print(f"BLEU-1        : {sum(bleu_scores['1']) / len(bleu_scores['1']):.4f}")
#     print(f"BLEU-2        : {sum(bleu_scores['2']) / len(bleu_scores['2']):.4f}")
#     print(f"BLEU-3        : {sum(bleu_scores['3']) / len(bleu_scores['3']):.4f}")
#     print(f"BLEU-4        : {sum(bleu_scores['4']) / len(bleu_scores['4']):.4f}")
#     print("-" * 40)
#     print(f"ROUGE-1 F1    : {sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']):.4f}")
#     print(f"ROUGE-2 F1    : {sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']):.4f}")
#     print(f"ROUGE-L F1    : {sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']):.4f}")
#     print("="*40)

#     # 儲存詳細預測結果
#     output_file = "evaluation_results.json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)
#     print(f"[INFO] Detailed results saved to {output_file}")

# if __name__ == "__main__":
#     main()
import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader

from videollava.model.builder import load_pretrained_model
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from videollava.conversation import conv_templates
from videollava.mm_utils import tokenizer_image_token

# 確保 nltk 資源已下載
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

@dataclass
class ModelArgs:
    image_tower: Optional[str] = None
    video_tower: Optional[str] = None
    audio_tower: Optional[str] = None
    mm_vision_select_layer: int = -2
    mm_vision_select_feature: str = "patch"
    pretrain_mm_mlp_adapter: Optional[str] = None
    mm_projector_type: str = "linear"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Video-LLaVA model on Shrimp Dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model or LoRA weights")
    parser.add_argument("--model-base", type=str, default=None, help="Base model path if using LoRA")
    parser.add_argument("--eval-json", type=str, required=True, help="Path to the validation json file")
    parser.add_argument("--av-root", type=str, default="", help="Root directory for audio/video files")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    # ✅ 新增參數：控制背景讀取資料的執行緒數量
    parser.add_argument("--num-workers", type=int, default=4, help="Number of background workers for data loading")
    return parser.parse_args()

def extract_qa(conversations):
    """從 conversations 中提取 human 的問題與 gpt 的標準答案"""
    question, answer = "", ""
    for msg in conversations:
        if msg["from"] in ["human", "user"] and not question:
            question = msg["value"].replace("<image>", "").replace("<video>", "").replace("<audio>", "").strip()
            if "\n" in question:
                question = question.split("\n", 1)[1].strip()
        elif msg["from"] in ["gpt", "assistant"] and not answer:
            # 先取得完整的答案
            full_answer = msg["value"].strip()
            # 使用 '\n' 切割字串，並用 [-1] 取得最後一個段落
            answer = full_answer.split('\n')[-1].strip()
            
    return question, answer

# -----------------------------
# 🚀 新增：資料集與預讀邏輯
# -----------------------------
class EvalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def eval_collate_fn_builder(tokenizer, model_config, video_processor, audio_processor, args):
    def collate_fn(batch):
        # 評估階段為了安全起見，我們強制 batch_size=1，交由 workers 並行讀取
        item = batch[0]
        try:
            video_path = os.path.join(args.av_root, item["video"]) if args.av_root else item["video"]
            audio_path = os.path.join(args.av_root, item["audio"]) if args.av_root else item["audio"]
            
            question, ground_truth = extract_qa(item["conversations"])
            
            # 處理影像張量 (在 CPU 上進行)
            video_tensor = video_processor(video_path, return_tensors="pt")["pixel_values"][0]
            if video_tensor.shape[0] != 3 and video_tensor.shape[1] == 3:
                video_tensor = video_tensor.permute(1, 0, 2, 3).contiguous()
            
            T = video_tensor.shape[1]
            if T > args.max_frames:
                video_tensor = video_tensor[:, :args.max_frames, :, :]
            elif T < args.max_frames:
                pad = video_tensor[:, -1:, :, :].repeat(1, args.max_frames - T, 1, 1)
                video_tensor = torch.cat([video_tensor, pad], dim=1)
                
            # 處理音訊張量 (在 CPU 上進行)
            audio_tensor = audio_processor(audio_path, return_tensors="pt")
            if isinstance(audio_tensor, dict):
                audio_tensor = list(audio_tensor.values())[0]

            # 準備 Prompt 與文字張量
            conv = conv_templates[args.conv_mode].copy()
            special_tokens = [DEFAULT_IMAGE_TOKEN] * args.max_frames
            if getattr(model_config, "mm_use_im_start_end", False):
                mm_prefix = "".join([DEFAULT_IM_START_TOKEN + t + DEFAULT_IM_END_TOKEN for t in special_tokens])
            else:
                mm_prefix = "".join(special_tokens)
                
            user_msg = mm_prefix + "\n" + question
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            
            return {
                "valid": True,
                "id": item["id"],
                "ground_truth": ground_truth,
                "input_ids": input_ids,
                "video_tensor": video_tensor,
                "audio_tensor": audio_tensor
            }
            
        except Exception as e:
            print(f"[WARN] Error loading media for {item.get('id', 'unknown')}: {e}")
            return {"valid": False}
            
    return collate_fn


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入模型與處理器
    print("[INFO] Loading model...")
    tokenizer, model, processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=args.model_path,
        load_8bit=False,
        load_4bit=False
    )
    
    # 強制掛載並初始化 Audio Tower
    audio_tower_name = getattr(model.config, "mm_audio_tower", "LanguageBind/LanguageBind_Audio")
    image_tower_name = getattr(model.config, "mm_image_tower", "LanguageBind/LanguageBind_Image")
    video_tower_name = getattr(model.config, "mm_video_tower", "LanguageBind/LanguageBind_Video_merge")
    
    model_args = ModelArgs(
        image_tower=image_tower_name,
        video_tower=video_tower_name,
        audio_tower=audio_tower_name,
    )
    model.model.initialize_vision_modules(model_args)
    
    # 🛡️ 最佳化：確保新掛載的模組移至正確的硬體與 bf16 精度
    model.to(device=device, dtype=torch.bfloat16)

    model.eval()
    
    vt = model.get_video_tower()
    at = model.get_audio_tower()
    if not vt.is_loaded:
        vt.load_model()
    if not at.is_loaded:
        at.load_model()
        
    video_processor = vt.video_processor
    audio_processor = at.audio_processor
    
    # 初始化 Scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    results = []
    bleu_scores = {"1": [], "2": [], "3": [], "4": []}
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    # 2. 讀取驗證集資料
    with open(args.eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    print(f"[INFO] Total evaluation samples: {len(eval_data)}")
    
    # 🚀 最佳化：建立 DataLoader 將資料讀取背景化
    dataset = EvalDataset(eval_data)
    collate_fn = eval_collate_fn_builder(tokenizer, model.config, video_processor, audio_processor, args)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1, # 推論時保持 1 筆避免維度錯位，提速靠 background workers
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 3. 推論與計算指標
    for batch in tqdm(dataloader, desc="Evaluating"):
        if not batch["valid"]:
            continue
            
        # 將 CPU 準備好的 Tensor 直接搬上 GPU 並轉為 bf16
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        video_tensor = batch["video_tensor"].to(device, dtype=torch.bfloat16)
        audio_tensor = batch["audio_tensor"].to(device, dtype=torch.bfloat16)
        ground_truth = batch["ground_truth"]
        
        # 執行生成
        with torch.inference_mode():
            # 動態設定生成參數，避免出現 temperature 與 do_sample 衝突的警告
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "use_cache": True
            }
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False
                
            output_ids = model.generate(
                input_ids=input_ids,
                images={"vision": [video_tensor], "audio": audio_tensor},
                **gen_kwargs
            )
            
        # 處理輸出文字
        prediction = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 4. 計算 Metrics
        ref_tokens = nltk.word_tokenize(ground_truth.lower())
        pred_tokens = nltk.word_tokenize(prediction.lower())
        
        bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        bleu_scores["1"].append(bleu1)
        bleu_scores["2"].append(bleu2)
        bleu_scores["3"].append(bleu3)
        bleu_scores["4"].append(bleu4)
        
        rouge_res = scorer.score(ground_truth, prediction)
        rouge_scores["rouge1"].append(rouge_res["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(rouge_res["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(rouge_res["rougeL"].fmeasure)
        
        results.append({
            "id": batch["id"],
            "ground_truth": ground_truth,
            "prediction": prediction,
            "metrics": {
                "bleu4": bleu4,
                "rougeL_f1": rouge_res["rougeL"].fmeasure
            }
        })

    # 5. 輸出總平均
    if len(results) > 0:
        print("\n" + "="*40)
        print("📋 Evaluation Results (Average)")
        print("="*40)
        print(f"Total Samples : {len(results)}")
        print(f"BLEU-1        : {sum(bleu_scores['1']) / len(bleu_scores['1']):.4f}")
        print(f"BLEU-2        : {sum(bleu_scores['2']) / len(bleu_scores['2']):.4f}")
        print(f"BLEU-3        : {sum(bleu_scores['3']) / len(bleu_scores['3']):.4f}")
        print(f"BLEU-4        : {sum(bleu_scores['4']) / len(bleu_scores['4']):.4f}")
        print("-" * 40)
        print(f"ROUGE-1 F1    : {sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1']):.4f}")
        print(f"ROUGE-2 F1    : {sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2']):.4f}")
        print(f"ROUGE-L F1    : {sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL']):.4f}")
        print("="*40)

        # 儲存詳細預測結果
        output_file = "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Detailed results saved to {output_file}")
    else:
        print("[WARN] No results to compute metrics. Please check data loading.")

if __name__ == "__main__":
    main()
