import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = ""
LORA_ADAPTER = ""

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

model = PeftModel.from_pretrained(
    model,
    LORA_ADAPTER,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
    # peft[head] must include this or else it does not mod params
    is_trainable=True,
)

# merge peft finetune into base model
model.merge_and_unload()

# use safetensors by default
model.save_pretrained("./hf_ckpt", max_shard_size="1GB", safe_serialization=True)
