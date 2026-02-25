# -*- coding: utf-8 -*-
"""
This scripts fine-tunes a language model using the unsloth library, which provides efficient training techniques such as 4-bit quantization and LoRA adapters.
"""
# unsloth must be imported first
import unsloth
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import weave # wandb: Use W&B Weave for improved LLM call tracing.
import argparse
import json
import os
import torch
import wandb
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer

def train(args):
    """
    Main function to execute the model loading, data preparation,
    and training pipeline.
    """
    run = wandb.init(
        config={
            **vars(args),                            # logs ALL argparse fields
        },
    )
    # also pin it into the run summary for quick visibility
    wandb.run.summary["dataset"] = args.dataset
    wandb.run.summary["dataset_split"] = args.split
    # 1. Load the model and tokenizer with 4-bit quantization
    print("🚀 Loading the model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,        # Specify sequence length for long context
        load_in_4bit=True,          # Enable 4-bit quantization for memory efficiency
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )
    if args.chat_template_path:
        with open(args.chat_template_path) as f:
            chat_template = f.read()
        print(f"🧩 Loaded custom chat template from {args.chat_template_path}")
        tokenizer.chat_template = chat_template
    # 2. Add LoRA adapters to the model for Parameter-Efficient Fine-Tuning (PEFT)
    print("🔧 Configuring the model with LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,                       # LoRA rank
        lora_alpha=args.lora_alpha,              # LoRA alpha
        lora_dropout=args.lora_dropout,             # Set dropout to 0 for optimization
        bias="none",                # Bias is not used for optimization
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        use_gradient_checkpointing="unsloth", # Use unsloth's efficient checkpointing
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # 3. Load and prepare the dataset
    print("📚 Loading and preparing the dataset...")
    dataset = []
    if os.path.exists(args.dataset):
        with open(args.dataset) as f:
            for line in f:
                dataset.append(json.loads(line))
        dataset = Dataset.from_list(dataset)
        wandb.run.summary["dataset"] = args.dataset
        print(f"Loaded dataset from local path: {args.dataset}")
    else:
        dataset = load_dataset(args.dataset, split=args.split)
        wandb.run.summary["dataset"] = args.dataset
        wandb.run.summary["dataset_split"] = args.split
    if args.sample_size:
        dataset = dataset.select(range(args.sample_size))
        wandb.run.summary["sample_size"] = args.sample_size
        print(f"Using a sample size of {args.sample_size} from the dataset.")

    def formatting_prompts_func(examples, 
                                think_token_start: str = "<think>",
                                think_token_end: str = "</think>"):
        """Applies the chat template to the conversations."""
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        texts = [text.replace("<think>\n", think_token_start).replace("\n</think>\n", think_token_end) for text in texts] 
        return {"text": texts}
    
    dataset = dataset.map(lambda examples: formatting_prompts_func(
        examples,
        think_token_start=args.think_token_start,
        think_token_end=args.think_token_end
    ), batched=True)
    print("Dataset formatting complete. Showing the first example:")
    print(dataset['text'][0])
    # 4. Configure and initialize the SFTTrainer
    print("⚙️ Setting up the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_length=args.max_seq_length,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # Mimics a larger batch size
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,             # A single pass over the dataset
            learning_rate=args.learning_rate,              # Starting learning rate
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="wandb",              # Change to "none" if not using WandB
        ),
    )

    # 5. Optimize training to focus only on assistant's responses
    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part=args.instruction_part,
    #     response_part=args.response_part
    # )

    # 6. Display GPU memory statistics before training
    print("📊 Displaying GPU memory stats before training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # 7. Start the training process
    print("\n🎉 Starting the training process!")
    trainer_stats = trainer.train()

    print("\n✅ Training finished successfully!")
    print("Training stats:", trainer_stats)
    trainer.save_model(args.output_dir)
