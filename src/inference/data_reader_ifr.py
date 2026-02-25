"""
Module for reading IFEval data in JSONL format for inference.
"""

import jsonlines
from typing import List, Dict

def read_ifr_data(data_file: str, inject_think_token: bool, think_token: str, sample_size: int = None) -> List[Dict]:
    """
    Reads IFEval data from a JSONL file.
    Args:
        data_file: Path to the JSONL file.
        sample_size: If set, only return up to this many examples.
    Returns:
        List of data items (dicts).
    """
    if inject_think_token:
        raise NotImplementedError("inject_think_token is not implemented yet.")
        # Need to change the logic to plain text prompts instead of chat format because of problems with the chat template and vllm.
        # I'm not using this in the final experiments.

    dataset = []
    with jsonlines.open(data_file) as reader:
        for item in reader:
            dataset.append(item)
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    prompts = [item['prompt'] for item in dataset]
    # Convert prompts to chat format. 
    # The dataset ifeval_lrm has the instructions targergting LRM models
    conversations = []
    for prompt in prompts:
        conversation = []
        conversation.append({"role": "user", "content": prompt})
        if inject_think_token:
            # add a final assistant turn to start the generation
            conversation.append({"role": "assistant", "content": think_token})
        conversations.append(conversation)
    
    return conversations
