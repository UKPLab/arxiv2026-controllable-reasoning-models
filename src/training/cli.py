import argparse
from .unsloth_training import train

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth + TRL")

    # dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/Multilingual-Thinking",
        help="Dataset name or local path"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset name or local path"
    )
    # model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path or HF repo for the model"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--instruction_part",
        type=str,
        default="<|start|>user<|message|>",
        help="How to identify the instruction part of the prompt assistant_loss_only"
    )
    parser.add_argument(
        "--response_part",
        type=str,
        default="<|start|>assistant<|channel|>analysis<|message|>",
        help="How to identify the response part of the prompt for assistant_loss_only"
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        help="Path to the chat template file",
    )
    # lora parameters
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout",
    )
    # training parameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Number of samples to use from the dataset",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to save the LoRA adapter",
    )
    parser.add_argument(
        "--think_token_start",
        type=str,
        default="<think>\n",
        help="Token to indicate the start of a reasoning trace",
    )
    parser.add_argument(
        "--think_token_end",
        type=str,
        default="\n</think>\n",
        help="Token to indicate the end of a reasoning trace",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
