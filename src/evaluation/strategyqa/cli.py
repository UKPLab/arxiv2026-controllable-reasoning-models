import argparse
from .evaluation import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Password Eval Benchmark.")
    parser.add_argument(
        "--thinking-path",
        type=str,
        required=True,
        help="Path to the file containing the model's thinking part.",
    )
    parser.add_argument(
        "--final-answer-path",
        type=str,
        required=True,
        help="Path to the file containing the model's final answers.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )

    return parser.parse_args()

def main():
    '''
    Main function to run evaluation from command line.
    Need to define HF_TOKEN in env variables.
    '''
    args = parse_args()
    evaluate(
        thinking_path=args.thinking_path,
        final_ans_path=args.final_answer_path,
        split=args.split,
    )

if __name__ == "__main__":
    main()
