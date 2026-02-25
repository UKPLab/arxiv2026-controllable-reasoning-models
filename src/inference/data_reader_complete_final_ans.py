import json
import os
from transformers import PreTrainedTokenizerBase

def read_data(folder_path: str, tokenizer: PreTrainedTokenizerBase, sample_size: int = None):
    conversations_path = os.path.join(folder_path, "responses_conversations.jsonl")
    thinkings_path = os.path.join(folder_path, "responses_thinking.jsonl")
    with open(conversations_path) as f:
        conversations = [json.loads(line) for line in f]
        # Includes system prompt (if any), and user prompt.
    with open(thinkings_path) as f:
        thinkings = [json.loads(line) for line in f]
        # Only includes assistant's thinking response.


    # chat = [
    #     {"role": "user", "content": "What is 1+1?"},
    #     {"role": "assistant", "content": '<think>\nIt seems pretty simple, 1+1 is 2.\n</think>\n\n'},
    # ]

    # # formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message=True, add_generation_prompt=False)
    # formatted_prompt = tokenizer.apply_chat_template(chat[:1], tokenize=False, continue_final_message=False, add_generation_prompt=True)
    # prompts = [formatted_prompt + f"{chat[1]['content']}"]

    prompts = []
    for conv, thinking in zip(conversations, thinkings):
        formatted_prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, continue_final_message=False, add_generation_prompt=True
        )
        prompts.append(formatted_prompt + f"{thinking['response']}\n</think>\n\n")

    if sample_size is not None:
        prompts = prompts[:sample_size]
        conversations = conversations[:sample_size]
        thinkings = thinkings[:sample_size]
    
    return prompts, conversations, thinkings

    complete_final_answers = []
    for conv, thinking in zip(conversations, thinkings):
        assistant_response = thinking["response"] + "</think>\n"
        conv.append({"role": "assistant", "content": assistant_response})
        complete_final_answers.append(conv)

    if sample_size is not None:
        complete_final_answers = complete_final_answers[:sample_size]
    return complete_final_answers
