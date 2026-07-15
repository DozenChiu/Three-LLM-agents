import os
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "molmo2-4b")

DATA_PATH = os.path.join(
    ROOT_DIR,
    "molmo_data",
    "custom",
    "shrimp",
    "hf_shrimp_train.json",
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "runs",
    "shrimp_lora_full_0708",
)

MAX_SAMPLES = None   # 正式版用全資料
BF16 = True

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 1


class ShrimpHFDataset(Dataset):
    def __init__(self, json_path: str, max_samples=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class ShrimpCollator:
    processor: Any
    vocab_size: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        video_paths = []

        for ex in batch:
            msgs = ex["messages"]

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": ex["video"]},
                        {"type": "text", "text": msgs[0]["content"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": msgs[1]["content"]},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            video_paths.append(ex["video"])

        model_inputs = self.processor(
            text=texts,
            videos=video_paths,
            return_tensors="pt",
            padding=True,
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        if "attention_mask" in model_inputs:
            labels[model_inputs["attention_mask"] == 0] = -100

        labels[(labels < 0) | (labels >= self.vocab_size)] = -100

        model_inputs["labels"] = labels
        return model_inputs


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    start_time = time.time()

    print("loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    print("processor OK")

    print("loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map=None,
    )
    print("model OK")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "att_proj",
            "ff_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = ShrimpHFDataset(DATA_PATH, max_samples=MAX_SAMPLES)
    vocab_size = model.config.vocab_size
    print("vocab_size:", vocab_size)

    num_samples = len(dataset)
    steps_per_epoch = math.ceil(
        num_samples / (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    )
    total_train_steps = steps_per_epoch * NUM_TRAIN_EPOCHS

    print("num_samples:", num_samples)
    print("per_device_train_batch_size:", PER_DEVICE_TRAIN_BATCH_SIZE)
    print("gradient_accumulation_steps:", GRADIENT_ACCUMULATION_STEPS)
    print("num_train_epochs:", NUM_TRAIN_EPOCHS)
    print("estimated_steps_per_epoch:", steps_per_epoch)
    print("estimated_total_train_steps:", total_train_steps)
    print("output_dir:", OUT_DIR)

    collator = ShrimpCollator(
        processor=processor,
        vocab_size=vocab_size,
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-4,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=BF16,
        fp16=not BF16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("starting training...")
    trainer.train()
    trainer.save_model(OUT_DIR)

    try:
        processor.save_pretrained(OUT_DIR)
    except AttributeError as e:
        print("processor.save_pretrained skipped:", e)
        try:
            processor.tokenizer.save_pretrained(OUT_DIR)
            print("tokenizer saved instead")
        except Exception as e2:
            print("tokenizer save also skipped:", e2)

    elapsed = time.time() - start_time
    print("saved to", OUT_DIR)
    print("total elapsed:", format_seconds(elapsed))


if __name__ == "__main__":
    main()
