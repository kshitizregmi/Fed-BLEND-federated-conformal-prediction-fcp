
import math, json, os, numpy as np, pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

DATA_ROOT = "scienceqa_sft_clients_with_images"
LORA_ADAPTER_PATH = "flower_qwen_lora_outputs/federated_adapter_final"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ALPHA = 0.10
LETTERS = ["A", "B", "C", "D"]
N_BINS = 2
MIN_CALIB_PER_BIN = 20

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

client_data = {}
for name in sorted(os.listdir(DATA_ROOT)):
    if name.startswith("client_"):
        cid = int(name.split("_")[1])
        client_dir = os.path.join(DATA_ROOT, name)
        client_data[cid] = {
            "validation": read_jsonl(os.path.join(client_dir, "validation.jsonl")),
            "calibration": read_jsonl(os.path.join(client_dir, "calibration.jsonl")),
            "test": read_jsonl(os.path.join(client_dir, "test.jsonl")),
        }

for cid, splits in client_data.items():
    print(f"CLIENT {cid}:", {k: len(v) for k, v in splits.items()})

# ---------- Load model with LoRA adapter ----------
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model.eval()

# ---------- Helpers ----------
def get_gold(row):
    return row["messages"][1]["content"].strip()[0]

def get_text_and_image(row):
    content = row["messages"][0]["content"]
    text_parts, image_path = [], None
    for item in content:
        if item["type"] == "text":
            text_parts.append(item["text"])
        elif item["type"] == "image":
            image_path = item.get("image")
    text = "\n".join(text_parts)
    img = None
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
    return text, img

def make_message(row):
    text, img = get_text_and_image(row)
    content = []
    if img is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}], img

def get_abcd_features(row):
    tokenizer = processor.tokenizer
    messages, img = make_message(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if img is None:
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    next_logits = out.logits[0, -1, :]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    logits = next_logits[choice_ids].float().detach().cpu().numpy()
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    margin = float(sorted_probs[0] - sorted_probs[1])
    max_prob = float(sorted_probs[0])
    energy = float(-np.log(np.sum(np.exp(logits))))
    return probs, logits, entropy, margin, max_prob, energy

rows = []
for cid, splits in client_data.items():
    print(f"Scoring client {cid}")
    for split_name in ["validation", "calibration", "test"]:
        for idx, row in enumerate(splits[split_name]):
            probs, logits, entropy, margin, max_prob, energy = get_abcd_features(row)
            gold = get_gold(row)
            gold_idx = LETTERS.index(gold)
            pred_idx = int(np.argmax(probs))
            pred = LETTERS[pred_idx]
            rows.append({
                "client_id": cid, "split": split_name, "idx": idx,
                "gold": gold, "pred": pred, "correct": int(pred == gold),
                "score_correct": 1.0 - float(probs[gold_idx]),
                "p_A": float(probs[0]), "p_B": float(probs[1]),
                "p_C": float(probs[2]), "p_D": float(probs[3]),
                "z_A": float(logits[0]), "z_B": float(logits[1]),
                "z_C": float(logits[2]), "z_D": float(logits[3]),
                "entropy": entropy, "margin": margin,
                "max_prob": max_prob, "energy": energy,
                "category": row.get("category"),
                "subject": row.get("subject"),
                "topic": row.get("topic"),
                "has_image": row.get("has_image"),
            })

df = pd.DataFrame(rows)
df["dscp_bin"] = 0  # placeholder, will be recalculated in downstream scripts

df.to_csv("fed_dscp_scored_examples.csv", index=False)
print("\nSaved: fed_dscp_scored_examples.csv")