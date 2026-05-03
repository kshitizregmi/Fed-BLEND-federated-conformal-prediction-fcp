import os
import json
import gc
import subprocess
from pathlib import Path
from collections import OrderedDict

# Stop Ray from using uv's runtime-env hook.
# Setting it to an empty string is more reliable than popping it,
# because `uv run` can re-inject the var and Ray's check is
# `bool(os.environ.get("RAY_RUNTIME_ENV_HOOK"))` -> bool("") is False.
os.environ["RAY_RUNTIME_ENV_HOOK"] = ""
# Belt-and-suspenders: also remove any uv-specific markers
os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
os.environ["RAY_RUNTIME_ENV_HOOK"] = ""

# ============================================================
# GPU AUTO-SELECTION BEFORE TORCH IMPORT
# ============================================================

MIN_FREE_GPU_GB = 20.0


def query_free_gpus(min_free_gb=20.0):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )

        good = []
        for line in out.strip().splitlines():
            idx_str, free_mb_str = line.split(",")
            idx = int(idx_str.strip())
            free_mb = int(free_mb_str.strip())
            free_gb = free_mb / 1024.0

            if free_gb >= min_free_gb:
                good.append(idx)

        return good

    except Exception as e:
        print("Could not query GPUs with nvidia-smi:", e)
        return []


SELECTED_GPUS = query_free_gpus(MIN_FREE_GPU_GB)

if len(SELECTED_GPUS) == 0:
    raise RuntimeError(
        f"No GPU has at least {MIN_FREE_GPU_GB} GB free. "
        "Free GPU memory or lower MIN_FREE_GPU_GB."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, SELECTED_GPUS))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Selected physical GPUs:", SELECTED_GPUS)
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

# --- New Flower API (Flower 1.13+) ---
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

DATA_ROOT = str(SCRIPT_DIR / "scienceqa_sft_clients_with_images")
OUTPUT_DIR = str(SCRIPT_DIR / "flower_qwen_lora_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "local_adapters"), exist_ok=True)


NUM_CLIENTS = 5
NUM_ROUNDS = 10

LOCAL_EPOCHS = 1
LOCAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0

MAX_LOCAL_STEPS = 80
MAX_EVAL_STEPS = 20

USE_4BIT = True
USE_GRAD_CHECKPOINTING = True

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

TOTAL_CPUS = os.cpu_count() or 8
NUM_AVAILABLE_GPUS = len(SELECTED_GPUS)

CLIENT_GPUS = 1.0
MAX_PARALLEL_CLIENTS = max(1, NUM_AVAILABLE_GPUS)

CLIENT_CPUS = max(2, min(12, TOTAL_CPUS // MAX_PARALLEL_CLIENTS))

print("Total CPUs:", TOTAL_CPUS)
print("Selected GPU count:", NUM_AVAILABLE_GPUS)
print("CPUs per Flower client:", CLIENT_CPUS)
print("GPUs per Flower client:", CLIENT_GPUS)
print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_client_rows(client_id, split):
    path = os.path.join(DATA_ROOT, f"client_{client_id}", f"{split}.jsonl")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return read_jsonl(path)


class ScienceQASFTDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def get_gold_text(row):
    ans = row["messages"][1]["content"].strip()
    return ans[0]


def get_user_text_and_image(row):
    user_content = row["messages"][0]["content"]

    text_parts = []
    image_path = None

    for item in user_content:
        if item["type"] == "text":
            text_parts.append(item["text"])
        elif item["type"] == "image":
            image_path = item.get("image", None)

    text = "\n".join(text_parts)

    image = None
    if image_path is not None and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")

    return text, image


def make_features(row, processor, device):
    user_text, image = get_user_text_and_image(row)
    answer = get_gold_text(row)

    user_content = []

    if image is not None:
        user_content.append({"type": "image"})

    user_content.append({"type": "text", "text": user_text})

    prompt_messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    full_messages = [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if image is None:
        prompt_inputs = processor(
            text=[prompt_text],
            return_tensors="pt",
        )

        full_inputs = processor(
            text=[full_text],
            return_tensors="pt",
        )

    else:
        prompt_inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        )

        full_inputs = processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
        )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels[:, :prompt_len] = -100

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    full_inputs["labels"] = labels

    full_inputs = full_inputs.to(device)

    return full_inputs


def make_collate_fn(processor, device):
    def collate_fn(batch):
        if len(batch) != 1:
            raise ValueError("Use LOCAL_BATCH_SIZE = 1.")
        return make_features(batch[0], processor, device)

    return collate_fn

def compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_processor():
    return AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )


def make_lora_model(training=True):
    dtype = compute_dtype()

    quant_config = None

    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quant_config,
    )

    model.config.use_cache = False

    if USE_4BIT:


        # In make_lora_model(), after prepare_model_for_kbit_training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=USE_GRAD_CHECKPOINTING,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # add this
        )

        # model = prepare_model_for_kbit_training(
        #     model,
        #     use_gradient_checkpointing=USE_GRAD_CHECKPOINTING,
        # )
    elif training and USE_GRAD_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    if training:
        model.train()
    else:
        model.eval()

    return model


def get_model_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lora_state(model):
    state = get_peft_model_state_dict(model)
    return OrderedDict((k, state[k]) for k in sorted(state.keys()))


def get_lora_keys_and_arrays(model):
    state = get_lora_state(model)

    keys = list(state.keys())
    arrays = [
        state[k].detach().cpu().to(torch.float32).numpy()
        for k in keys
    ]

    return keys, arrays


def get_lora_arrays(model, keys):
    state = get_lora_state(model)

    return [
        state[k].detach().cpu().to(torch.float32).numpy()
        for k in keys
    ]


def set_lora_arrays(model, keys, arrays):
    current_state = get_lora_state(model)
    new_state = {}

    for key, arr in zip(keys, arrays):
        target = current_state[key]

        tensor = torch.as_tensor(
            arr,
            dtype=target.dtype,
            device=target.device,
        )

        new_state[key] = tensor

    set_peft_model_state_dict(
        model,
        new_state,
        adapter_name="default",
    )


print("\nInitializing global LoRA parameters...")

_init_processor = make_processor()
_init_model = make_lora_model(training=False)

LORA_KEYS, INITIAL_ARRAYS = get_lora_keys_and_arrays(_init_model)

print("Number of LoRA tensors:", len(LORA_KEYS))
print(
    "Number of LoRA parameters:",
    f"{sum(np.prod(x.shape) for x in INITIAL_ARRAYS):,}",
)

_init_processor.save_pretrained(os.path.join(OUTPUT_DIR, "processor"))
_init_model.save_pretrained(os.path.join(OUTPUT_DIR, "initial_adapter"))

del _init_model
del _init_processor

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def train_one_client(model, processor, rows):
    device = get_model_device(model)

    dataset = ScienceQASFTDataset(rows)

    loader = DataLoader(
        dataset,
        batch_size=LOCAL_BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(processor, device),
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    num_steps = 0
    opt_steps = 0

    for _ in range(LOCAL_EPOCHS):
        for batch in loader:
            if MAX_LOCAL_STEPS is not None and num_steps >= MAX_LOCAL_STEPS:
                break

            outputs = model(**batch)

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM_STEPS
            num_steps += 1

            if num_steps % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

        if MAX_LOCAL_STEPS is not None and num_steps >= MAX_LOCAL_STEPS:
            break

    if num_steps > 0 and num_steps % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            1.0,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_steps += 1

    avg_loss = total_loss / max(num_steps, 1)

    return avg_loss, num_steps, opt_steps


@torch.no_grad()
def eval_one_client(model, processor, rows):
    device = get_model_device(model)

    dataset = ScienceQASFTDataset(rows)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(processor, device),
        num_workers=0,
        pin_memory=False,
    )

    model.eval()

    total_loss = 0.0
    num_steps = 0

    for batch in loader:
        if MAX_EVAL_STEPS is not None and num_steps >= MAX_EVAL_STEPS:
            break

        outputs = model(**batch)
        total_loss += float(outputs.loss.detach().cpu())
        num_steps += 1

    avg_loss = total_loss / max(num_steps, 1)

    return avg_loss, num_steps


class QwenFlowerClient(NumPyClient):
    def __init__(self, client_id):
        self.client_id = int(client_id)

        print(f"[Client {self.client_id}] Loading model")

        self.processor = make_processor()
        self.model = make_lora_model(training=True)

        self.train_rows = load_client_rows(self.client_id, "train")
        self.val_rows = load_client_rows(self.client_id, "validation")

        print(
            f"[Client {self.client_id}] "
            f"train={len(self.train_rows)} validation={len(self.val_rows)}"
        )

    def get_parameters(self, config):
        return get_lora_arrays(self.model, LORA_KEYS)

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))

        print(f"[Client {self.client_id}] Fit round {server_round}")

        set_lora_arrays(self.model, LORA_KEYS, parameters)

        train_loss, train_steps, opt_steps = train_one_client(
            self.model,
            self.processor,
            self.train_rows,
        )

        local_adapter_dir = os.path.join(
            OUTPUT_DIR,
            "local_adapters",
            f"client_{self.client_id}",
            f"round_{server_round}",
        )

        os.makedirs(local_adapter_dir, exist_ok=True)

        self.model.save_pretrained(local_adapter_dir)

        updated_parameters = get_lora_arrays(self.model, LORA_KEYS)

        print(
            f"[Client {self.client_id}] "
            f"round={server_round} "
            f"loss={train_loss:.4f} "
            f"steps={train_steps} "
            f"opt_steps={opt_steps} "
            f"saved={local_adapter_dir}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return updated_parameters, len(self.train_rows), {
            "train_loss": float(train_loss),
            "train_steps": int(train_steps),
            "opt_steps": int(opt_steps),
        }

    def evaluate(self, parameters, config):
        set_lora_arrays(self.model, LORA_KEYS, parameters)

        val_loss, val_steps = eval_one_client(
            self.model,
            self.processor,
            self.val_rows,
        )

        print(
            f"[Client {self.client_id}] "
            f"eval_loss={val_loss:.4f} "
            f"eval_steps={val_steps}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(val_loss), len(self.val_rows), {
            "val_loss": float(val_loss),
            "val_steps": int(val_steps),
        }


# --- New-API client_fn: takes a Context, partition id comes from node_config ---
def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    return QwenFlowerClient(partition_id).to_client()


# Build the ClientApp once at module level
client_app = ClientApp(client_fn=client_fn)


def weighted_average_fit_metrics(metrics):
    total = sum(num_examples for num_examples, _ in metrics)

    if total == 0:
        return {}

    return {
        "train_loss": sum(n * m.get("train_loss", 0.0) for n, m in metrics) / total,
        "train_steps": sum(n * m.get("train_steps", 0.0) for n, m in metrics) / total,
        "opt_steps": sum(n * m.get("opt_steps", 0.0) for n, m in metrics) / total,
    }


def weighted_average_eval_metrics(metrics):
    total = sum(num_examples for num_examples, _ in metrics)

    if total == 0:
        return {}

    return {
        "val_loss": sum(n * m.get("val_loss", 0.0) for n, m in metrics) / total
    }


class SaveFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        self.latest_ndarrays = None
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:
            self.latest_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            round_npz = os.path.join(
                OUTPUT_DIR,
                f"global_lora_round_{server_round}.npz",
            )

            np.savez(
                round_npz,
                **{k: v for k, v in zip(LORA_KEYS, self.latest_ndarrays)},
            )

            print(f"[Server] Saved aggregated LoRA tensors: {round_npz}")

        return aggregated_parameters, aggregated_metrics


def fit_config(server_round: int):
    return {
        "server_round": server_round,
    }


# We need a handle to the strategy after the simulation finishes,
# so we keep it on a module-level holder.
strategy_holder = {"strategy": None}


def server_fn(context: Context) -> ServerAppComponents:
    initial_parameters = ndarrays_to_parameters(INITIAL_ARRAYS)

    strategy = SaveFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average_fit_metrics,
        evaluate_metrics_aggregation_fn=weighted_average_eval_metrics,
    )

    # Hold a reference so we can read latest_ndarrays after simulation.
    strategy_holder["strategy"] = strategy

    config = ServerConfig(num_rounds=NUM_ROUNDS)

    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)

# ============================================================
# RAY / FLOWER RUN (new run_simulation API)
# ============================================================

# IMPORTANT: Do NOT include 'runtime_env' inside init_args.
# Flower's raybackend.py already passes runtime_env to ray.init() on its
# own. Providing it here too triggers:
#   TypeError: ray.init() got multiple values for keyword argument 'runtime_env'
# Run with: uv run --no-project python <script>.py


import ray

# Wipe every env var that could make Ray auto-connect to a foreign cluster
for _var in [
    "RAY_ADDRESS",
    "RAY_RUNTIME_ENV_HOOK",
    "UV_PROJECT_ENVIRONMENT",
]:
    os.environ.pop(_var, None)

# Force Ray to always start fresh (never connect to existing cluster)
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

# Shutdown if somehow already initialized in this process
if ray.is_initialized():
    ray.shutdown()

# Start our own clean Ray cluster
# ray.init(
#     address=None,           # Never connect to existing — always start fresh
#     num_cpus=TOTAL_CPUS,
#     num_gpus=NUM_AVAILABLE_GPUS,
#     include_dashboard=False,
#     ignore_reinit_error=False,  # False so we catch double-init bugs early
#     logging_level="error",      # Suppress Ray's noisy warnings
#     _temp_dir=os.path.join(OUTPUT_DIR, "ray_tmp"),  # Isolated temp dir per run
# )

ray.init(
    address=None,
    num_cpus=TOTAL_CPUS,
    num_gpus=NUM_AVAILABLE_GPUS,
    include_dashboard=False,
    ignore_reinit_error=False,
    logging_level="error",
    _temp_dir="/tmp/ray_flwr",
    runtime_env={
        "working_dir": str(SCRIPT_DIR),
        "excludes": [
            # large data and output folders
            "scienceqa_sft_clients_with_images",
            "flower_qwen_lora_outputs",
            # any other large files
            "*.npz",
            "*.pt",
            "*.bin",
            "*.safetensors",
            "*.csv",
            "*.pdf",
            "*.png",
            "*.log",
        ],
    },
)


print(f"Ray initialized: {ray.cluster_resources()}")


backend_config = {
    "client_resources": {
        "num_cpus": CLIENT_CPUS,
        "num_gpus": CLIENT_GPUS,
    },
    "init_args": {
        # num_cpus / num_gpus intentionally omitted —
        # Ray is already running, passing them here would crash
        "include_dashboard": False,
        "ignore_reinit_error": True,
    },
}

print("\nStarting Flower simulation (new run_simulation API)")
print("backend_config:", backend_config)

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)








# backend_config = {
#     "client_resources": {
#         "num_cpus": CLIENT_CPUS,
#         "num_gpus": CLIENT_GPUS,
#     },
#     "init_args": {
#         "num_cpus": TOTAL_CPUS,
#         "num_gpus": NUM_AVAILABLE_GPUS,
#         "include_dashboard": False,
#         "ignore_reinit_error": True,
#     },
# }

# print("\nStarting Flower simulation (new run_simulation API)")
# print("backend_config:", backend_config)

# run_simulation(
#     server_app=server_app,
#     client_app=client_app,
#     num_supernodes=NUM_CLIENTS,
#     backend_config=backend_config,
# )

# print("\nFlower simulation complete.")


strategy = strategy_holder["strategy"]

if strategy is None or strategy.latest_ndarrays is None:
    raise RuntimeError("No aggregated parameters found. Training failed.")

final_adapter_dir = os.path.join(OUTPUT_DIR, "federated_adapter_final")

print("\nSaving final federated LoRA adapter:", final_adapter_dir)

final_processor = make_processor()
final_model = make_lora_model(training=False)

set_lora_arrays(final_model, LORA_KEYS, strategy.latest_ndarrays)

os.makedirs(final_adapter_dir, exist_ok=True)

final_model.save_pretrained(final_adapter_dir)
final_processor.save_pretrained(final_adapter_dir)

del final_model
del final_processor

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nDONE")
print("Final federated adapter:", final_adapter_dir)
print("Local adapters:", os.path.join(OUTPUT_DIR, "local_adapters"))

print("\nTesting adapter load...")

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=compute_dtype(),
    device_map="auto",
)

loaded_model = PeftModel.from_pretrained(
    base_model,
    final_adapter_dir,
)

loaded_model.eval()

print("Adapter loaded successfully.")
print("Use this path later:")
print("LORA_ADAPTER_PATH =", final_adapter_dir)