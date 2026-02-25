import argparse
import os
import sys
import json
import re

from .constraint_registry import INSTRUCTION_DICT


def evaluate(data_path, thinking_path, final_ans_path, print_stats=False, output_path_name="evaluation_metrics.json"):

    # Load model outputs
    responses_thinking = []
    with open(thinking_path, "r") as f:
        for line in f:
            responses_thinking.append(json.loads(line))

    responses_final_ans = []
    with open(final_ans_path, "r") as f:
        for line in f:
            responses_final_ans.append(json.loads(line))

    # count empty thinking
    count_empty_thinking = sum(1 for r in responses_thinking if len(r['response'].strip()) <10)
    # count empty final answers
    count_empty_final = sum(1 for r in responses_final_ans if len(r['response'].strip()) < 10)
    if count_empty_thinking > 0:
        print(f"Warning: {count_empty_thinking} out of {len(responses_thinking)} thinking responses are empty or too short.")
    if count_empty_final > 0:
        print(f"Warning: {count_empty_final} out of {len(responses_final_ans)} final responses are empty or too short.")

    metrics = math_if_evaluate(
        data_path=data_path,
        responses_path=thinking_path,
        print_stats=print_stats,
    )
    final_metrics = math_if_evaluate(
        data_path=data_path,
        responses_path=final_ans_path,
        print_stats=print_stats,
    )
    avg_if = (metrics["loose_accuracy"] + final_metrics["loose_accuracy"]) / 2
    ans_acc = evaluate_answer_accuracy(
        data_path=data_path,
        responses_path=final_ans_path,
        print_stats=print_stats,
    )
    # Save metrics to a json file
    output_metrics = {
        "thinking": metrics,
        "final_answer": final_metrics,
        "answer_accuracy": ans_acc,
        "avg_thinking_if_utility": (metrics["loose_accuracy"] + ans_acc) / 2,
        "avg_final_answer_if_utility": (final_metrics["loose_accuracy"] + ans_acc) / 2,
        "avg_if": avg_if,
        "avg_if_utility": (avg_if + ans_acc) / 2,
        "empty_thinking_responses": count_empty_thinking,
        "empty_final_answers": count_empty_final,
    }
    output_path = os.path.join(
        os.path.dirname(final_ans_path),
        output_path_name,
    )
    with open(output_path, "w") as f:
        json.dump(output_metrics, f, indent=4)
    print(f"Saved evaluation metrics to {output_path}")
    print(f"{output_metrics['thinking']['loose_accuracy']*100:.2f}\t{output_metrics['final_answer']['loose_accuracy']*100:.2f}\t{output_metrics['avg_if']*100:.2f}\t{output_metrics['answer_accuracy']*100:.2f}\t{output_metrics['empty_thinking_responses']}\t{output_metrics['empty_final_answers']}")

    # save printing
    with open(os.path.join(
        os.path.dirname(final_ans_path),
        "evaluation_summary.txt",
    ), "w") as f:
        f.write("Summary Metrics (Tab-separated):\n")
        # columns
        f.write("Thinking_IF\tFinal_Answer_IF\tAvg_IF\tAnswer_Accuracy\tEmpty_Thinking_Responses\tEmpty_Final_Answers\n")
        f.write(f"{output_metrics['thinking']['loose_accuracy']*100:.2f}\t{output_metrics['final_answer']['loose_accuracy']*100:.2f}\t{output_metrics['avg_if']*100:.2f}\t{output_metrics['answer_accuracy']*100:.2f}\t{output_metrics['empty_thinking_responses']}\t{output_metrics['empty_final_answers']}\n")
    return output_metrics

def evaluate_answer_accuracy(data_path, responses_path, print_stats=False):
    '''
    Evaluates the answer accuracy by comparing extracted answers from responses
    to the ground truth labels in the data.
    Args:
        data_path (str): Path to the data file containing ground truth answers.
        responses_path (str): Path to the responses file containing model outputs.
        print_stats (bool): Whether to print the accuracy statistics.
    Returns:
        float: The computed answer accuracy.
    '''
    # Open labels
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    list_labels = [item['answer'] for item in data]
    # Open responses
    with open(responses_path, "r") as f:
        responses = [json.loads(line) for line in f.readlines()]
    list_ans = extract_answers(responses)
    # compute accuracy
    correct = 0
    for pred, label in zip(list_ans, list_labels):
        if pred == label:
            correct += 1
    accuracy = correct / len(list_labels)
    if print_stats:
        print(f"Answer Accuracy: {accuracy:.4f}")
    return accuracy

def math_if_evaluate(data_path, responses_path, print_stats=False):
    strict = []
    loose = []
    # correct = []


    for line1,line2 in zip(open(responses_path).readlines(), open(data_path).readlines()):
        try:
            hypothesis = json.loads(line1)["output"]
        except:
            hypothesis = json.loads(line1)["response"]
        if isinstance(hypothesis,list):
            hypothesis = hypothesis[0]
        
        data = json.loads(line2)
        if not ("noconstraint" in responses_path):
            is_follow_list = test_instruction_following_strict(
                data["instruction_id_list"],
                hypothesis,
                data["kwargs"],
                data["question"],
            )
            strict.append(all(is_follow_list))
            loose.append(sum(is_follow_list)/len(is_follow_list))
        else:
            # only place holder
            strict.append(1)
            loose.append(1)

        # if compute_score(hypothesis, data['answer'])[0]:
        #     correct.append(1)
        # else:
        #     correct.append(0)
    
    metrics =  {"strict_accuracy": sum(strict)/len(strict),
                "loose_accuracy": sum(loose)/len(loose)}
    if print_stats:
        print("Evaluation Results:")
        print(json.dumps(metrics, indent=4))
    return metrics


def test_instruction_following_strict(
    instruction_id_list,
    response,
    parameters,
    prompt,
):
    """Tests response to see if instructions are followed."""

    is_following_list = []
    for index, instruction_id in enumerate(instruction_id_list):
        try:
            instruction_cls = INSTRUCTION_DICT[instruction_id]
        except:
            import pdb
            pdb.set_trace()
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.  
        if parameters[index]:
            kwargs = {n: p for n, p in parameters[index].items() if p}
        else:
            kwargs = {}
        instruction.build_description(**kwargs)
        args = instruction.get_constraint_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        try:
            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
        except:
            import pdb
            pdb.set_trace()

    return is_following_list


def extract_boxed_answer(text):
    '''
    Extracts the last occurrence of a boxed answer from the given text.
    Args:
        text (str): The input text containing potential boxed answers.
    Returns:
        str or None: The last boxed answer if found, otherwise None.
    '''
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if len(matches) == 0:
        return None
    return matches[-1]

def extract_last_number(text):
    '''
    Extracts the last number (integer or decimal) from the given text.
    Args:
        text (str): The input text containing potential numbers.
    Returns:
        str or None: The last number if found, otherwise None.
    '''
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) == 0:
        return None
    return numbers[-1]

def extract_answer(response):
    '''
    Extracts the answer from the response text by first looking for a boxed answer,
    and if not found, extracting the last number. If neither is found, returns None.
    Args:
        response (str): The input response text.
    Returns:
        str or None: The extracted answer if found, otherwise None.
    '''
    ans = extract_boxed_answer(response)
    if ans is None:
        ans = extract_last_number(response)
    return ans

def extract_answers(responses):
    '''
    Extracts answers from a list of response dictionaries.
    Args:
        responses (list): A list of dictionaries, each containing a 'response' key.
    Returns:
        list: A list of extracted answers.
    '''
    list_ans = []
    for response in responses:
        list_ans.append(extract_answer(response['response']))
    return list_ans