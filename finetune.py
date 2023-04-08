import os
import sys
from typing import List

import fire
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter


def train(
        # model/data params
        logging_steps: int = 1,
        load_8bit: bool = True,  # for 7B, we can load as fp16
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
        lr_scheduler_type: str = "linear",
        optimizer: str = "adamw_torch",
        learning_rate: float = 3e-4,
        cutoff_len: int = 1024,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
        padding_side: str = "left",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"logging_steps: {logging_steps}\n"
            f"load_8bit: {load_8bit}\n"
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
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template}\n"
            f"padding_side: {padding_side}\n"

        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # transformer head resolves to LlamaTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    tokenizer.pad_token_id = (
         0  # unk. we want this to be different from the eos token
    )

    # check lit-llama discussion
    if padding_side != "right" and padding_side != "left":
        print(f"padding_side only allow allow 'left' or 'right' entered: {padding_side}")
        exit(1)

    tokenizer.padding_side = padding_side  # Allow batched inference

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

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    from datasets import concatenate_datasets, load_dataset

    datas = []

    # support multi json files
    if train_data_json is not None:
        data_json = load_dataset("json", data_files=train_data_json)
        datas.append(data_json["train"])
        print("\n\ndata_json size: " + str(len(data_json["train"])))

    if train_data_set is not None:
        # limit=200000
        temp = load_dataset(train_data_set) # , split=f'train[:{limit}]')
        print("\n\ndata_set size: " + str(len(temp["train"])))
        datas.append(temp["train"])

    # merge all datas
    data = {'train': concatenate_datasets(datas)}
    print("\n\nCombined dataset size: " + str(len(data["train"])))

    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    else:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
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

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            lr_scheduler_type=lr_scheduler_type,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim=optimizer,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_limit,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
