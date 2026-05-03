# !pip install -q datasets scikit-learn transformers accelerate qwen-vl-utils pillow

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from PIL import Image
import random
import math
import os
import json
import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NUM_CLIENTS = 5
SEED = 42

TRAIN_FRAC = 0.70
MIN_TRAIN_PER_CATEGORY = 20
MIN_TOTAL_PER_CATEGORY = math.ceil(MIN_TRAIN_PER_CATEGORY / TRAIN_FRAC)

OUT_DIR = "scienceqa_sft_clients_with_images"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LETTERS = ["A", "B", "C", "D"]

rng = random.Random(SEED)

print("Minimum total per category kept:", MIN_TOTAL_PER_CATEGORY)


ds = load_dataset("derek-thomas/ScienceQA")

def keep_4choice_mcq(ex):
    return (
        ex.get("task") == "closed choice"
        and ex.get("question") is not None
        and isinstance(ex.get("choices"), list)
        and len(ex["choices"]) == 4
        and ex.get("answer") in [0, 1, 2, 3]
        and ex.get("category") is not None
        and ex.get("subject") is not None
        and ex.get("topic") is not None
    )

all_examples = []

for split in ["train", "validation", "test"]:
    filtered = ds[split].filter(keep_4choice_mcq)

    for ex in filtered:
        ex = dict(ex)
        ex["category"] = str(ex["category"]).strip()
        ex["subject"] = str(ex["subject"]).strip()
        ex["topic"] = str(ex["topic"]).strip()
        all_examples.append(ex)

print("Total filtered:", len(all_examples))

# Drop categories too small to give >=20 train examples
cat_counts = Counter(ex["category"] for ex in all_examples)

kept_examples = [
    ex for ex in all_examples
    if cat_counts[ex["category"]] >= MIN_TOTAL_PER_CATEGORY
]

dropped_categories = [
    c for c, n in cat_counts.items()
    if n < MIN_TOTAL_PER_CATEGORY
]

print("Kept examples:", len(kept_examples))
print("Dropped categories:", dropped_categories)

# Build non-IID clients
# Whole categories assigned to clients.
# This avoids random 2-3 category fragments inside clients.
by_category = defaultdict(list)

for ex in kept_examples:
    by_category[ex["category"]].append(ex)

categories = list(by_category.keys())
rng.shuffle(categories)

clients = [[] for _ in range(NUM_CLIENTS)]

for i, category in enumerate(categories):
    client_id = i % NUM_CLIENTS
    clients[client_id].extend(by_category[category])

for client in clients:
    rng.shuffle(client)

# Split each client IID by category
# train/validation/calibration/test have same category distribution per client
client_splits = []

for cid, client in enumerate(clients):
    if len(client) == 0:
        client_splits.append({
            "train": [],
            "validation": [],
            "calibration": [],
            "test": [],
        })
        continue

    labels = [x["category"] for x in client]

    train, temp = train_test_split(
        client,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    val, temp = train_test_split(
        temp,
        test_size=2/3,
        random_state=SEED,
        stratify=[x["category"] for x in temp],
    )

    calib, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        stratify=[x["category"] for x in temp],
    )

    client_splits.append({
        "train": train,
        "validation": val,
        "calibration": calib,
        "test": test,
    })

# Check split
for cid, splits in enumerate(client_splits):
    print(f"\nCLIENT {cid}")
    print("sizes:", {k: len(v) for k, v in splits.items()})

    train_counts = Counter(x["category"] for x in splits["train"])
    bad = {k: v for k, v in train_counts.items() if v < MIN_TRAIN_PER_CATEGORY}

    print("train categories below 20:", bad)
    print("train categories:", train_counts.most_common())
    print("validation categories:", Counter(x["category"] for x in splits["validation"]))
    print("calibration categories:", Counter(x["category"] for x in splits["calibration"]))
    print("test categories:", Counter(x["category"] for x in splits["test"]))

# Save SFT JSONL with images
os.makedirs(OUT_DIR, exist_ok=True)

def save_image_if_exists(ex, path):
    img = ex.get("image", None)

    if img is None:
        return None

    if isinstance(img, Image.Image):
        img.save(path)
        return path

    return None

def to_sft_example(ex, image_path=None):
    choices = ex["choices"]
    ans_idx = int(ex["answer"])

    prompt = (
        "Answer the multiple-choice science question.\n\n"
        f"Question: {ex['question']}\n\n"
        "Choices:\n"
        + "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices)])
        + "\n\nReturn only the correct option letter: A, B, C, or D."
    )

    answer = f"{LETTERS[ans_idx]}"

    user_content = []

    if image_path is not None:
        user_content.append({
            "type": "image",
            "image": image_path,
        })

    user_content.append({
        "type": "text",
        "text": prompt,
    })

    return {
        "messages": [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "category": ex["category"],
        "subject": ex["subject"],
        "topic": ex["topic"],
        "answer": int(ex["answer"]),
        "has_image": image_path is not None,
    }

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

for client_id, splits in enumerate(client_splits):
    client_dir = os.path.join(OUT_DIR, f"client_{client_id}")
    image_dir = os.path.join(client_dir, "images")

    os.makedirs(client_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    for split_name, data in splits.items():
        rows = []

        for i, ex in enumerate(data):
            image_path = os.path.join(image_dir, f"{split_name}_{i}.png")
            saved_image_path = save_image_if_exists(ex, image_path)
            rows.append(to_sft_example(ex, saved_image_path))

        out_path = os.path.join(client_dir, f"{split_name}.jsonl")
        write_jsonl(out_path, rows)

print("\nSaved SFT data to:", OUT_DIR)

# Pick one client-0 train example for test inference
# Prefer image example. If none exists, use text-only example.
CLIENT_ID = 0

example = None

for ex in client_splits[CLIENT_ID]["train"]:
    if ex.get("image") is not None:
        example = ex
        break

if example is None:
    example = client_splits[CLIENT_ID]["train"][0]

print("\nSelected example:")
print("category:", example["category"])
print("subject:", example["subject"])
print("topic:", example["topic"])
print("has image:", example.get("image") is not None)
print("question:", example["question"])
print("choices:", example["choices"])
print("gold:", LETTERS[int(example["answer"])])

# Load VLM
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.eval()





def raw_ex_to_message(ex):
    prompt = (
        "Answer the multiple-choice science question.\n\n"
        f"Question: {ex['question']}\n\n"
        "Choices:\n"
        + "\n".join([f"{LETTERS[i]}. {choice}" for i, choice in enumerate(ex["choices"])])
        + "\n\nReturn only the correct option letter: A, B, C, or D."
    )

    content = []

    if ex.get("image") is not None:
        content.append({"type": "image"})

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]

messages = raw_ex_to_message(example)

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

img = example.get("image", None)

if img is None:
    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)
else:
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    ).to(model.device)


with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]

output = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\nMODEL OUTPUT:")
print(output)

print("\nGOLD:")
print(LETTERS[int(example["answer"])])

# A/B/C/D probabilities for the same example
def get_abcd_probs(model, processor, ex):
    tokenizer = processor.tokenizer
    messages = raw_ex_to_message(ex)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    img = ex.get("image", None)

    if img is None:
        inputs = processor(
            text=[text],
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        ).to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    next_logits = out.logits[0, -1, :]

    choice_token_ids = []
    for c in LETTERS:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{c} tokenized into multiple tokens: {ids}")
        choice_token_ids.append(ids[0])

    choice_logits = next_logits[choice_token_ids]
    choice_probs = torch.softmax(choice_logits, dim=-1)

    return {
        LETTERS[i]: choice_probs[i].item()
        for i in range(4)
    }

abcd_probs = get_abcd_probs(model, processor, example)
pred = max(abcd_probs, key=abcd_probs.get)

print("\nA/B/C/D probabilities:")
print(abcd_probs)

print("\nPred:", pred)
print("Gold:", LETTERS[int(example["answer"])])