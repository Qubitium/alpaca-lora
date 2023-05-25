import os
from os.path import exists, join, isdir
import sys
import typing
from typing import Optional, Dict, Sequence
from typing import List

import fire
import torch
import bitsandbytes as bnb
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict, PeftModel,
)
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.prompter import Prompter

torch.backends.cuda.matmul.allow_tf32 = True

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def print_trainable_parameters(bits, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")


def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train(
        # model/data params
        logging_steps: int = 1,
        lora: bool = True,  # if use lora/peft
        bits: int = 8,  # train in 16bit, 8bit, 4 bit
        bf16: bool = False,
        fp16: bool = True,
        # tf32: bool = True,
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 0.3,
        base_model: str = "",  # the only required argument
        train_data_json: List[str] = None,  # json files
        train_data_set: str = None,  # dataset
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        warmup_ratio: float = 0.03,
        eval_steps: int = 100,
        save_steps: int = 100,
        save_limit: int = 16,
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 4,
        lr_scheduler_type: str = "cosine",
        optimizer: str = "paged_adamw_32bit",
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        val_set_ratio: float = 0.05,
        # lora hyperparams
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.00,
        lora_target_modules: List[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: typing.Union[str, bool] = None,  # either training checkpoint or final adapter
        prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
        padding_side: str = "right",
):
    # only one option bf16 or fp16 can be activated
    if bf16:
        fp16 = False

    device_map = {"": "cuda:0"}

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}

    gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # warn user if value is 0 and auto fix
    if gradient_accumulation_steps == 0:
        print(
            "-------------- WARNING --------------\n"
            "Calculated gradient_accumulation_steps is 0. Check your WORLD_SIZE, batch_size and micro_batch_size.\n"
        )

        batch_size = world_size * micro_batch_size
        gradient_accumulation_steps = 1
        print(
            f"Auto fixed gradient_accumulation_steps to 1 and changed batch_size to {batch_size}",
            f"using formula: word_size:{world_size} x micro_batch_size:{micro_batch_size} = batch_size:{batch_size}"
        )

    # Ensure value is min 1
    gradient_accumulation_steps = max(gradient_accumulation_steps, 1)

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta/llama-7b-hf'"

    prompter = Prompter(prompt_template)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )

    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    if resume_from_checkpoint:
        os.environ["WANDB_RESUME"] = "Allow"

    # transformer head resolves to LlamaTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
        padding_side=padding_side,
    )

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_GB - 2}GB"

    print(f"device_map: {device_map}\n")

    if bits == 8:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # max_memory=max_memory,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
    elif bits == 4:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # max_memory=max_memory,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            # doc https://github.com/huggingface/transformers/pull/23479/files#diff
            # -4333c40134efe7287ccda3bdd11e266b90a62629b933bf2cd15fa39cbf23b088R82
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
        )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    print(model)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    else:
        model.is_parallelizable = False
        model.model_parallel = False

    model.config.torch_dtype = (torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    from datasets import concatenate_datasets, load_dataset

    datas = []

    # support multi json files
    if train_data_json is not None:
        data_json = load_dataset("json", data_files=train_data_json)
        datas.append(data_json["train"])
        print("\ndata_json size: " + str(len(data_json["train"])))

    if train_data_set is not None:
        dataset_list = train_data_set.split(',')
        for d in dataset_list:
            print(f"\nLoading dataset: {d}")
            temp = load_dataset(d)  # , split=f'train[:{limit}]')
            print("\ndata_set size: " + str(len(temp["train"])))
            datas.append(temp["train"])

    # merge all datas
    data = {'train': concatenate_datasets(datas)}
    print("\nCombined dataset size: " + str(len(data["train"])))

    if lora:
        lora_target_modules = find_all_linear_names(bits, model)
        print(f"modules: {lora_target_modules}\n")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if resume_from_checkpoint is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(resume_from_checkpoint, 'adapter_model'))
            for name, p in model.named_parameters():
                if 'lora' in name:
                    print(name, p.sum())
        else:
            print(f'adding LoRA modules...')
            model = get_peft_model(model, config)

        model = get_peft_model(model, config)

        if gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"logging_steps: {logging_steps}\n"
            f"lora: {lora}\n"
            f"bits: {bits}\n"
            f"bf16: {bf16}\n"
            f"fp16: {fp16}\n"
            # f"tf32: {tf32}\n"
            f"gradient_checkpointing: {gradient_checkpointing}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"max_grad_norm: {max_grad_norm}\n"
            f"base_model: {base_model}\n"
            f"train_data_json: {train_data_json}\n"
            f"train_data_set: {train_data_set}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"lr_scheduler_type: {lr_scheduler_type}\n"
            f"optimizer: {optimizer}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"save_steps: {save_steps}\n"
            f"save_limit: {save_limit}\n"
            f"eval_steps: {eval_steps}\n"
            f"warmup_ratio: {warmup_ratio}\n"
            f"val_set_ratio: {val_set_ratio}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template}\n"
            f"padding_side: {padding_side}\n"

        )

    print_trainable_parameters(bits, model)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    val_set_size = 0  # default to 0
    if val_set_ratio > 0:
        val_set_size = int(len(data["train"]) * val_set_ratio)
        print(f"\nCalculated val_set_size: {val_set_size}")
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=learning_rate,
            bf16=bf16,
            fp16=fp16,
            logging_steps=logging_steps,
            optim=optimizer,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_limit,
            # load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=(8 if bf16 or fp16 else None), return_tensors="pt"
        ),
    )

    # NOT WORKING, result in empty model checkpoints trainer.add_callback(SavePeftModelCallback)

    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    if bits < 16:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    model = torch.compile(model)

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
