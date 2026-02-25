from datasets import load_dataset
import json
from sklearn.metrics import accuracy_score, f1_score
import os
from openai import OpenAI
from tqdm.auto import trange
import langid
import numpy as np
import importlib.resources



def evaluate(thinking_path, final_ans_path, split):
    dataset = load_dataset("haritzpuerto/strategyqa-icot-think", split=split)

    client = OpenAI(
        # increase default timeout to 15 minutes (from 10 minutes)
        timeout=900.0,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    thinking_responses = []
    with open(thinking_path, "r") as f:
        for line in f:
            thinking_responses.append(json.loads(line))
    final_answer_responses = []
    with open(final_ans_path, "r") as f:
        for line in f:
            final_answer_responses.append(json.loads(line))

    # instruction following evaluation
    if_acc, cost1 = instruction_following_evaluation(
        dataset, thinking_responses, final_answer_responses, client
    )
    print(f"Instruction Following Accuracy: {if_acc:.2f}")
    # reasoning performance evaluation
    reasoning_acc, cost2 = evaluate_accuracy(dataset, final_answer_responses, client)
    print(f"Reasoning Accuracy: {reasoning_acc:.2f}")
    dir_path = os.path.dirname(final_ans_path)
    evaluation_path = os.path.join(dir_path, "evaluation.json")
    print(f"Saving evaluation results to {evaluation_path}")
    with open(evaluation_path, "w") as f:
        json.dump({"IF_acc": if_acc, "reasoning_acc": reasoning_acc, "Total Cost": cost1 + cost2}, f, indent=4)

def evaluate_accuracy(dataset, final_answer_responses, client):
    list_results = []
    pbar = trange(len(dataset))
    total_cost = 0
    for i in pbar:
        question = dataset[i]["question"]
        answer = final_answer_responses[i]["response"]
        extracted_answer, cost = extract_answer(question, answer, client)
        extracted_answer = extracted_answer.lower()
        total_cost += cost
        if "yes" in extracted_answer:
            list_results.append(1)
        elif "no" in extracted_answer:
            list_results.append(0)
        else:
            print("Error in response:", extracted_answer)
        pbar.set_description(f"Current cost: ${total_cost:.6f}")
    # print("Reasoning Accuracy:", np.mean(list_results))
    return np.mean(list_results) * 100, total_cost


def extract_answer(question, answer, client):
    system_prompt = "The user provides a question and an answer. The question is always a boolean question, i.e., the answers can be yes or no. The answer provided by the user may be long, non-english, and wrong, that's fine. Your task is to extract 'yes' or 'no' from the answer. If the answer is not clear, you can make your best guess. Respond with 'Yes' or 'No' only."
    try:
        response = client.with_options(timeout=900.0).responses.create(
            model="gpt-5-nano-2025-08-07",
            instructions=f"{system_prompt}",
            input= f"Question:{question}\nAnswer:{answer}.",
            service_tier="flex",
        )
    except Exception as e:
        print("Error in response:", e)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        return "Yes", 0.0  # default to yes in case of error
    cost = response.usage.input_tokens * 0.025/1e6 + response.usage.output_tokens * 0.2/1e6
    return response.output_text, cost

def instruction_following_evaluation(
    dataset, thinking_responses, final_answer_responses, client
):
    list_results = []
    pbar = trange(len(dataset))
    total_cost = 0
    for idx in pbar:
        # for idx in trange(3):
        thinking_response = thinking_responses[idx]["response"]
        final_answer_response = final_answer_responses[idx]["response"]
        intr_eval = dataset[idx]["intr_eval"]
        instruction_type = dataset[idx]["instruction_type"]
        if thinking_response == "":
            # The model didn't generate a propoer thinking response, so it didn't follow instructions.
            list_results.append(0)
            continue

        if instruction_type == "bilingual_reasoning":
            thinking_lang = intr_eval.split("->")[0].strip()
            final_lang = intr_eval.split("->")[1].strip()
            thinking_eval = lang_classification(thinking_response, thinking_lang)
            final_eval = lang_classification(final_answer_response, final_lang)
            list_results.append((thinking_eval + final_eval) / 2)
        else:
            try:
                response = client.with_options(timeout=900.0).responses.create(
                    model="gpt-5-nano-2025-08-07",
                    instructions=f"{intr_eval}. Say 'Yes' or 'No' only.",
                    input=f"{thinking_response}",
                    service_tier="flex",
                )
            except Exception as e:
                print("Error in response:", e)
                print(f"Problem in index {idx}, skipping...")
                print(f"Thinking response: {thinking_response}")
                print(f"Instruction eval response: {intr_eval}")
                # There was a problem with the response, so we count it as a failure to follow instructions.
                list_results.append(0)
                continue
            cost = response.usage.input_tokens * 0.025/1e6 + response.usage.output_tokens * 0.2/1e6
            total_cost += cost
            if "yes" in response.output_text.lower():
                list_results.append(1)
            elif "no" in response.output_text.lower():
                list_results.append(0)
            else:
                print("Error in response:", response.choices[0].message.content)
        pbar.set_description(f"Current cost: ${total_cost:.6f}")
    # print("IF Accuracy:", np.mean(list_results) * 100)
    return np.mean(list_results) * 100, total_cost


def lang_classification(text, label):
    label2code = {
        "Spanish": "es",
        "English": "en",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
    }
    lang, confidence = langid.classify(text)
    return int(lang == label2code[label])