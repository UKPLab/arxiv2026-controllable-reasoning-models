"""
Module for running inference on a dataset using vLLM.
"""
import weave
import copy
import logging
import os
from typing import List, Dict, Optional
import jsonlines
import json
from tqdm import tqdm
import time
from vllm import LLM
import wandb
from .data_reader_ifr import read_ifr_data
from .data_reader_hf import read_hf_data
from .data_reader_complete_final_ans import read_data
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from types import SimpleNamespace

# Set up logging
logger = logging.getLogger(__name__)


def run_inference(args: dict) -> None:
    """
    Run inference on a dataset with vLLM.
    """
    try:
        # Pop arguments not used by LLM
        max_tokens = args.pop("max_tokens")
        temperature = args.pop("temperature")
        top_p = args.pop("top_p", None)
        top_k = args.pop("top_k", None)
        min_p = args.pop("min_p", None)
        seed = args.pop("seed", None)
        output_file = args.pop("output_file")
        data_file = args.pop("data_file")
        sample_size = args.pop("sample_size")
        batch_size = args.pop("batch_size")
        dataset_type = args.pop("dataset", "ifr")
        lora_path = args.pop("lora_path", None)
        system_prompt = args.pop("system_prompt", None)
        prompt_field = args.pop("prompt_field", "prompt")
        split = args.pop("split")
        inject_think_token = args.pop("inject_think_token", False) # not used
        think_token_start = args.pop("think_token_start")
        think_token_end = args.pop("think_token_end")
        incompatible_with_sys_prompt = args.pop("incompatible_with_sys_prompt", False)
        use_icl = args.pop("use_in_context_learning", False)
        use_hf_inference = args.pop("use_hf_inference", False)
        quantization = args.pop("quantization")
        prompt_injection = args.pop("prompt_injection", None)
        prompt_injection_file = args.pop("prompt_injection_file", None)
        staged_decoding = False

        # Resolve the prompt-injection attack text (inline string takes precedence
        # over a file). Only the `hf` reader applies it; for staged decoding the
        # injection is inherited from the stage-1 conversations file.
        if prompt_injection_file and not prompt_injection:
            with open(prompt_injection_file, "r") as f:
                prompt_injection = f.read().strip()
        if prompt_injection:
            logger.info(f"Prompt-injection attack ENABLED ({len(prompt_injection)} chars).")

        continue_generation = False # default is False. Will use chat api for generation
        if inject_think_token:
            raise NotImplementedError("inject_think_token is not implemented yet.")
        
        # Get wandb parameters from environment variables with fallback to args
        wandb_project = os.getenv(
            "WANDB_PROJECT", args.pop("wandb_project", None)
        )
        wandb_run_name = os.getenv("WANDB_RUN_NAME", args.pop("wandb_run_name", None))
        wandb_run_group = os.getenv(
            "WANDB_RUN_GROUP", args.pop("wandb_run_group", None)
        )

        # Initialize wandb only if project is set
        if wandb_project:
            wandb_config = {
                "model": args["model"],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "batch_size": batch_size,
                "sample_size": sample_size,
                **{k: v for k, v in args.items() if k != "model"},
            }

            # Add environment variables to config
            if os.getenv("SLURM_JOB_ID"):
                wandb_config["slurm_job_id"] = os.getenv("SLURM_JOB_ID")
            if os.getenv("CUDA_VISIBLE_DEVICES"):
                wandb_config["gpu_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")

            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                group=wandb_run_group,
                config=wandb_config,
            )
            logger.info(
                f"Initialized wandb logging (project: {wandb_project}, run: {wandb_run_name}, group: {wandb_run_group})"
            )

        # Create LLM instance
        logger.info(f"Initializing vLLM with model {args['model']}...")
        if "unsloth" not in args['model']:
            if quantization == "bitsandbytes":
                logger.info("Using bitsandbytes quantization for vLLM. Not an unsloth model.")
                llm = LLM(**args, enable_lora=lora_path is not None, max_lora_rank=64, quantization="bitsandbytes")
            else:
                logger.info("Not using bitsandbytes quantization for vLLM. Not an unsloth model.")
                llm = LLM(**args, enable_lora=lora_path is not None, max_lora_rank=64)
        else:
            llm = LLM(**args, enable_lora=lora_path is not None, max_lora_rank=64)
        logger.info("vLLM initialized.")

        # Load dataset using the appropriate reader
        if dataset_type == "ifr":
            logger.info(
                f"Loading IFEval dataset from {data_file} using IFEval reader..."
            )
            conversations = read_ifr_data(data_file, inject_think_token, think_token_start, sample_size)
        elif dataset_type == "hf":
            logger.info(f"Loading HF dataset from {data_file} using HF reader...")
            conversations = read_hf_data(data_file, use_icl, incompatible_with_sys_prompt, inject_think_token, think_token_start, sample_size, system_prompt_field=system_prompt, prompt_field=prompt_field, split=split, prompt_injection=prompt_injection)
        elif dataset_type == "complete_final_ans":
            logger.info(
                f"Loading Complete Final Answer dataset from {data_file} using Complete Final Answer reader..."
            )
            prompts, conversations, thinkings = read_data(data_file, llm.get_tokenizer(), sample_size)
            continue_generation = True
            staged_decoding = True

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        logger.info(f"Loaded {len(conversations)} conversations")

        
        # Create sampling params
        sampling_params = llm.get_default_sampling_params()
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        if temperature is not None:
            sampling_params.temperature = temperature
        if top_p is not None:
            sampling_params.top_p = top_p
        if top_k is not None:
            sampling_params.top_k = top_k
        if min_p is not None:
            sampling_params.min_p = min_p
        if seed is not None:
            sampling_params.seed = seed
        

        # Prepare output file
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )

        # Process dataset in batches
        logger.info(f"Starting inference (batch size: {batch_size})...")
        logger.info(f"Processing {len(conversations)} examples")

        if staged_decoding:
            all_outputs = generate_api(
                llm,
                sampling_params,
                prompts,
                batch_size,
                output_file,
                lora_path=lora_path,
                wandb=wandb,
            )
        else:
            all_outputs = chat_generation(
                llm,
                sampling_params,
                conversations,
                batch_size,
                output_file,
                lora_path=lora_path,
                wandb=wandb,
            )
        
        if staged_decoding:
            all_thinking, all_final_ans = stage_decoding_post_processing(all_outputs, conversations, thinkings)
        else:
            all_thinking, all_final_ans = full_generation_post_processing(all_outputs, think_token_end)

        # save to disk
        thinking_output_path = os.path.splitext(output_file)[0] + "_thinking.jsonl"
        final_ans_output_path = os.path.splitext(output_file)[0] + "_final_ans.jsonl"
        conversations_output_path = os.path.splitext(output_file)[0] + "_conversations.jsonl"
        with open(thinking_output_path, 'w') as f:
            for item in all_thinking:
                f.write(json.dumps(item) + "\n")
        with open(final_ans_output_path, 'w') as f:
            for item in all_final_ans:
                f.write(json.dumps(item) + "\n")
        with open(conversations_output_path, 'w') as f:
            for item in conversations:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Finished processing {len(conversations)} examples")
        logger.info(f"Results saved to {output_file}")


    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        raise



def get_max_model_len(llm, default=32768) -> int:
    """Best-effort retrieval of vLLM max context length."""
    try:
        return llm.llm_engine.model_config.max_model_len
    except Exception:
        return default

def prompt_token_len(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt).input_ids)

def compute_max_new_tokens(
    prompt_len: int,
    max_model_len: int,
    base_max_new: int | None,
    safety: int,
) -> int | None:
    """
    Returns:
      - int: allowed max_new_tokens for this prompt
      - None: prompt is too long to run at all (without truncation)
    """
    # Need room for prompt + (some) generation + safety
    if prompt_len >= (max_model_len - safety):
        return None

    available_new = (max_model_len - safety) - prompt_len
    if base_max_new is None:
        return available_new
    return min(base_max_new, available_new)

def make_error_result(prompt: str) -> dict:
    return {
        "prompt": prompt,
        "response": "",
    }

def build_sampling_params_list(
    base_sampling_params,
    max_new_tokens_list: list[int],
):
    """
    vLLM supports per-prompt SamplingParams by passing a list.
    We shallow-copy the base and override max_tokens.
    """
    params_list = []
    for mn in max_new_tokens_list:
        sp = copy.copy(base_sampling_params)
        sp.max_tokens = mn
        params_list.append(sp)
    return params_list

def scatter_outputs_into_results(batch_results, runnable_indices, subset_outputs):
    """Place vLLM outputs back into the original batch order."""
    for out_obj, j in zip(subset_outputs, runnable_indices):
        batch_results[j] = {
            "prompt": out_obj.prompt,
            "response": out_obj.outputs[0].text,
        }

def log_batch_metrics(wandb, batch_time, examples_processed, total_time, num_prompt_too_long):
    if wandb and wandb.run is not None:
        wandb.log({
            "batch_time": batch_time,
            "examples_processed": examples_processed,
            "total_time": total_time,
            "num_prompt_too_long": num_prompt_too_long,
        })

def generate_api(llm, sampling_params, conversations, batch_size, output_file, lora_path=None, wandb=None):
    """
    Continue the generation of the conversations while:
      - preserving input order
      - not truncating prompts
      - generating as many new tokens as fit in context for each prompt
      - emitting placeholder error rows for prompts that are too long to run
    """
    lora_request = None
    if lora_path:
        lora_request = LoRARequest("lora_adapter", 1, lora_path=lora_path)
        print(f"Using LoRA adapter from {lora_path}")

    tokenizer = llm.get_tokenizer()
    max_model_len = get_max_model_len(llm, default=32768)
    safety = 64  # buffer for special tokens / off-by-one

    base_max_new = getattr(sampling_params, "max_tokens", None)

    outputs = []
    total_time = 0

    with jsonlines.open(output_file, mode="w") as writer:
        for i in tqdm(range(0, len(conversations), batch_size)):
            batch_start_time = time.time()
            batch = conversations[i:i + batch_size]

            # Prepare placeholders to preserve order
            batch_results = [None] * len(batch)

            runnable_prompts = []
            runnable_indices = []
            runnable_max_new = []

            # Decide per prompt what we can generate (no truncation)
            for j, prompt in enumerate(batch):
                
                prompt_len = prompt_token_len(tokenizer, prompt)
                max_new_tokens = compute_max_new_tokens(prompt_len, max_model_len, base_max_new, safety)

                if max_new_tokens is None:
                    batch_results[j] = make_error_result(prompt)
                    continue
                runnable_prompts.append(prompt)
                runnable_indices.append(j)
                runnable_max_new.append(max_new_tokens)

            # Run only runnable prompts, then scatter back
            if runnable_prompts:
                runnable_params = build_sampling_params_list(sampling_params, runnable_max_new)
                subset_outputs = llm.generate(runnable_prompts, runnable_params, lora_request=lora_request)
                scatter_outputs_into_results(batch_results, runnable_indices, subset_outputs)

            batch_time = time.time() - batch_start_time
            total_time += batch_time

            log_batch_metrics(
                wandb=wandb,
                batch_time=batch_time,
                examples_processed=i + len(batch),
                total_time=total_time,
                num_prompt_too_long=len(batch)-len(runnable_prompts),
            )

            # Write + append in preserved order
            for r in batch_results:
                writer.write(r)
                outputs.append(r)

            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(conversations))} examples")

    if wandb and wandb.run is not None:
        wandb.log({
            "total_examples_processed": len(conversations),
            "total_time": total_time,
        })
        wandb.finish()

    return outputs


def chat_generation(llm, sampling_params, conversations, batch_size, output_file, lora_path=None, wandb=None):
    total_time = 0
    
    all_outputs = []
    with jsonlines.open(output_file, mode="w") as writer:
        for i in tqdm(range(0, len(conversations), batch_size)):
            batch_start_time = time.time()
            batch = conversations[i : i + batch_size]

            # Generate responses
            lora_request = None
            if lora_path:
                lora_request = LoRARequest("lora_adapter", 1, lora_path=lora_path)
                print(f"Using LoRA adapter from {lora_path}")
            try:
                outputs = llm.chat(
                    batch, sampling_params, use_tqdm=False, lora_request=lora_request,
                    continue_final_message=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.error(f"Error during LLM chat generation: {str(e)}")

                # Fallback: process each conversation one by one
                fallback_outputs = []
                for batch_idx, conv in enumerate(batch):
                    try:
                        single_out = llm.chat(
                            [conv],  # single conversation as a "batch" of size 1
                            sampling_params,
                            use_tqdm=False,
                            lora_request=lora_request,
                            continue_final_message=False,
                            add_generation_prompt=True,
                        )
                        # vLLM returns a list; take the first element
                        fallback_outputs.append(single_out[0])
                    except Exception as e_single:
                        logger.error(
                            f"Error during LLM chat generation for conversation index {i*batch_idx}: {str(e_single)}"
                        )
                        print(f"Conversation that failed: {conv}")

                        # Create a placeholder with the same interface: output.outputs[0].text
                        placeholder = SimpleNamespace(
                            outputs=[SimpleNamespace(text="")]
                        )
                        fallback_outputs.append(placeholder)

                outputs = fallback_outputs
            # Calculate metrics
            batch_time = time.time() - batch_start_time
            total_time += batch_time

            # Log metrics to wandb if initialized
            if wandb.run is not None:
                wandb.log(
                    {
                        "batch_time": batch_time,
                        "examples_processed": i + len(batch),
                        "total_time": total_time,
                    }
                )

            # Save results
            for conv, output in zip(batch, outputs):
                # For IFR, conv[1]['content'] is the prompt; for Cello, conv may be longer
                user_message = next(
                    (m["content"] for m in conv if m["role"] == "user"), None
                )
                generated_text = output.outputs[0].text
                # if the generation seems "weird," let's try one more time
                if len(generated_text) < 100:
                    logger.warning(f"Generated text seems too short. Retrying generation...\n\nPrompt: {user_message}\nGenerated: {generated_text}")
                    try:
                        retry_output = llm.chat(
                            [conv],
                            sampling_params,
                            use_tqdm=False,
                            lora_request=lora_request,
                            continue_final_message=False,
                            add_generation_prompt=True,
                        )
                        generated_text = retry_output[0].outputs[0].text
                        logger.warning(f"Retry successful. New generated text: {generated_text}")
                    except Exception as e_retry:
                        logger.error(f"Retry generation failed: {str(e_retry)}")
                
                result = {"prompt": user_message, "response": generated_text}
                writer.write(result)
                all_outputs.append(result)

            # Log progress
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {i + batch_size} examples")

    # Log final metrics if wandb is initialized
    if wandb.run is not None:
        wandb.log(
            {
                "total_examples_processed": len(conversations),
                "total_time": total_time,
            }
        )
        wandb.finish()
    return all_outputs

def full_generation_post_processing(all_outputs, think_token_end):
    '''
    Split the outputs into thinking and final answers based on the think_token_end.
    The split is done by looking for the last occurrence of think_token_end in the response.
    Args:
        all_outputs: List of outputs from the model, each containing 'prompt' and 'response'
        think_token_end: The token indicating the end of the thinking part (e.g., '</think>')
    Returns:
        all_thinking: List of dictionaries with 'prompt' and 'response' for the thinking part
        all_final_ans: List of dictionaries with 'prompt' and 'response' for the final answer part
    '''
    # post-process the outputs to get thinking and final answers
    all_thinking = []
    all_final_ans = []
    for output in all_outputs:
        response = output['response']
        # if there is </think> in response, we can split, there is reasoning and final answer
        if think_token_end in response:
            think = think_token_end.join(response.split(think_token_end)[:-1]).strip()
            # final answer is everything after </think>. Careful, there could be more than one occurrence of </think> (mainly because of model mistakes)
            # use last occurrence of </think> as starting point for the final answer
            # we use \n</think>\n because sometimes the model mention </think> as part of its reasoning
            final_ans = response.split(think_token_end)[-1].strip()
            all_thinking.append({'prompt': output['prompt'], 'response': think})
            all_final_ans.append({'prompt': output['prompt'], 'response': final_ans})
        else:
            # there is no </think>, we consider the whole response as reasoning, and final answer is empty
            think = response.strip()
            final_ans = ""
            all_thinking.append({'prompt': output['prompt'], 'response': think})
            all_final_ans.append({'prompt': output['prompt'], 'response': final_ans})

    return all_thinking, all_final_ans


def stage_decoding_post_processing(all_outputs, conversations, thinkings):
    '''
    The thinking part was generated in the first stage, and final answer in the second stage.
    So the thinking part is given in the conversations.
    The generated outputs only contain the final answers.
    '''
    # post-process the outputs to get thinking and final answers
    all_thinking = []
    all_final_ans = []
    for conv, thinking, response in zip(conversations, thinkings, all_outputs):
        # Get the system prompt and user prompt as the main prompt
        prompt = "" # Format Role: message\n
        for message in conv:
            prompt += f"{message['role']}: {message['content']}\n"
        # Get the last message as thinking part
        think = thinking['response']
        # Get the generated response as final answer
        final_ans = response['response'].strip()

        all_thinking.append({'prompt': prompt, 'response': think})
        all_final_ans.append({'prompt': prompt, 'response': final_ans})
        
    return all_thinking, all_final_ans