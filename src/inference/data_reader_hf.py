"""
Module for reading IFEval data in JSONL format for inference.
"""

from datasets import load_dataset
from typing import List, Dict
import os

def read_hf_data(hf_repository: str, use_icl:bool, incompatible_with_sys_prompt: bool, inject_think_token: bool, think_token: str, sample_size: int = None, system_prompt_field :str = None, prompt_field: str = 'prompt', split: str = 'test', prompt_injection: str = None) -> List[Dict]:
    """
    Reads a HF dataset
    Args:
        hf_repository: The Hugging Face repository name.
        sample_size: If set, only return up to this many examples.
        prompt_injection: If set, this text is appended to the final user turn of
            every conversation. Used to reproduce the reasoning-trace extraction
            attack of Green et al. (2025), where the injected instruction forces
            the model to reproduce its (hidden) reasoning trace in the visible
            final answer.
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
                # Try the absolute cluster path first (original setup), then fall
                # back to the repo-relative copy shipped under data/demonstrations.
                demo_candidates = [
                    f"/data/demonstrations/{hf_repository}/demonstration.txt",
                    os.path.join("data", "demonstrations", hf_repository, "demonstration.txt"),
                ]
                demo_path = next((p for p in demo_candidates if os.path.exists(p)), None)
                if demo_path is None:
                    raise FileNotFoundError(
                        f"ICL demonstration not found for '{hf_repository}'. Looked in: {demo_candidates}"
                    )
                with open(demo_path, "r") as f:
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
        # Prompt-injection attack (Green et al., 2025): append the injection to
        # the last user turn so it is the final instruction the model reads,
        # regardless of single- or multi-turn structure.
        if prompt_injection:
            for message in reversed(conversation):
                if message["role"] == "user":
                    message["content"] = message["content"] + "\n\n" + prompt_injection
                    break

        if inject_think_token:
            # add a final assistant turn to start the generation
            conversation.append({"role": "assistant", "content": think_token})

        # Add the conversation to the dataset of conversations
        conversations.append(conversation)

    return conversations
