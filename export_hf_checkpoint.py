import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F402

BASE_MODEL = ""
LORA_ADAPTER = ""

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

# save first weight for diff testing
first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
    #peft[head] must include this or else it does not mod params
    is_trainable=True,
)

lora_model.merge_and_unload()

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model.save_pretrained("./hf_ckpt", max_shard_size="500MB")
