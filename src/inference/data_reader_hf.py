"""
Module for reading IFEval data in JSONL format for inference.
"""

from datasets import load_dataset
from typing import List, Dict
import os

def read_hf_data(hf_repository: str, use_icl:bool, incompatible_with_sys_prompt: bool, inject_think_token: bool, think_token: str, sample_size: int = None, system_prompt_field :str = None, prompt_field: str = 'prompt', split: str = 'test') -> List[Dict]:
    """
    Reads a HF dataset
    Args:
        hf_repository: The Hugging Face repository name.
        sample_size: If set, only return up to this many examples.
    Returns:
        List of data items (dicts).
    """
    if inject_think_token:
        raise NotImplementedError("inject_think_token is not implemented yet")
        # Need to change the logic to plain text prompts instead of chat format because of problems with the chat template and vllm.
        # I'm not using this in the final experiments.

    print(f"Loading HF dataset from {hf_repository} with split {split}...")
    dataset = load_dataset(hf_repository, split=split)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    conversations = []
    for x in dataset:
        conversation = []
        user_prompt = x[prompt_field]
        # Add the system prompt
        if system_prompt_field:
            system_prompt = x[system_prompt_field]

            if use_icl:
                with open(f"/data/demonstrations/{hf_repository}/demonstration.txt", "r") as f:
                    demonstration = f.read().strip()
                system_prompt += "\n\n" + demonstration
            
            if not incompatible_with_sys_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            else:
                user_prompt = system_prompt + "\n\n" + user_prompt
        # Add the user prompt
        conversation.append({"role": "user", "content": user_prompt})
        
        # Add any assistant/user responses for multi-turn
        if "assistant_response" in x and x['assistant_response'].strip() != "":
            conversation.append({"role": "assistant", "content": x['assistant_response']})
        if "user_response" in x and x['user_response'].strip() != "":
            conversation.append({"role": "user", "content": x['user_response']})
        if inject_think_token:
            # add a final assistant turn to start the generation
            conversation.append({"role": "assistant", "content": think_token})
        
        # Add the conversation to the dataset of conversations
        conversations.append(conversation)

    return conversations
