"""Command line interface for the ifr package."""

import logging
import sys
from vllm import LLM
from vllm.utils.argparse_utils import FlexibleArgumentParser

from .inference import run_inference

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Entry point for the command line interface."""
    parser = FlexibleArgumentParser(description='Run LLaMA inference on IFEval dataset')
    
    # Add engine args manually instead of using EngineArgs.add_cli_args()
    engine_group = parser.add_argument_group("Engine arguments")
    engine_group.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                            help='Model name or path')
    engine_group.add_argument("--tensor-parallel-size", type=int, default=1,
                            help='Number of GPUs to use for tensor parallelism')
    engine_group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help='Fraction of GPU memory to use')
    engine_group.add_argument("--max-model-len", type=int, default=None,
                            help='Maximum sequence length')
    engine_group.add_argument("--dtype", type=str, default="auto",
                            help='Data type for model weights and activations')
    engine_group.add_argument("--trust-remote-code", action="store_true",
                            help='Trust remote code when loading models')
    engine_group.add_argument("--lora-path", type=str, default=None,
                            help='Path to LoRA weights to apply to the model (optional)')
    engine_group.add_argument("--use-hf-inference", action="store_true",
                            help='Whether to use Hugging Face inference instead of vLLM')
    
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=512,
                              help='Maximum length of generated responses')
    sampling_group.add_argument("--temperature", type=float, default=0.7,
                              help='Sampling temperature')
    sampling_group.add_argument("--top-p", type=float, default=0.9,
                              help='Top-p sampling parameter')
    sampling_group.add_argument("--top-k", type=int, default=None,
                              help='Top-k sampling parameter')
    sampling_group.add_argument("--min-p", type=float, default=None,
                              help='Minimum probability mass for nucleus sampling')
    sampling_group.add_argument("--seed", type=int, default=None,
                              help='Random seed for sampling (default: random)')
    
    # Add model params
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument("--inject-think-token", action="store_true",
                             help='Whether to inject a special think token for LRMs that use it (e.g., R1). Do not use for Qwen 3.')
    model_group.add_argument("--think-token-start", type=str, default="<think>",
                             help='Token to delimit the starting of a reasoning trajectory of an LRM. Default is <think> (R1)')
    model_group.add_argument("--think-token-end", type=str, default="</think>",
                             help='Token to delimit the ending of a reasoning trajectory of an LRM. Default is </think> (R1)')
    model_group.add_argument("--incompatible-with-sys-prompt", action="store_true",
                             help='Whether the model is INCOMPATIBLE with system prompts (e.g., R1). If set, system prompts from datasets will be included at the start of the user prompt instead.')
    model_group.add_argument("--quantization", type=str, default="bitsandbytes",
                             help='Quantization method to use (e.g., bitsandbytes, none)')
    # Add dataset params
    dataset_group = parser.add_argument_group("Dataset parameters")
    dataset_group.add_argument("--dataset", type=str, default="ifr",
                             help='Dataset type (default: ifr). Options: ifr, cello, hf, complete_final_ans')
    dataset_group.add_argument("--data-file", type=str, required=True,
                             help='Path to input JSONL file containing the IFEval dataset or the HF repository name if --dataset is hf. Folder path if --dataset is complete_final_ans')
    dataset_group.add_argument("--output-file", type=str, default='llama_ifeval_results.jsonl',
                             help='Path to output JSONL file')
    dataset_group.add_argument("--sample-size", type=int, default=None,
                             help='If set, only process this many examples (useful for testing)')
    dataset_group.add_argument("--batch-size", type=int, default=8,
                             help='Batch size for inference')
    dataset_group.add_argument("--system-prompt", type=str, default=None,
                             help='Field name for system prompt in HF dataset (if applicable)')
    dataset_group.add_argument("--prompt-field", type=str, default="prompt",
                             help='Field name for user prompt in HF dataset (if applicable)')
    dataset_group.add_argument("--split", type=str, default="test",
                             help='Dataset split to use for HF datasets (default: test)')
    dataset_group.add_argument("--use-in-context-learning", action="store_true",
                             help='Whether to use in-context learning (ICL) with demonstrations')
    
    
    args = parser.parse_args()
    
    try:
        if args.use_hf_inference:
            run_inference_hf(vars(args))
        else:
            run_inference(vars(args))
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()