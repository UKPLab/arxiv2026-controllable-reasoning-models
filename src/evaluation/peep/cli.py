import argparse
from .evaluation import evaluate_leaks, evaluate_utility

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Password Eval Benchmark.")
    parser.add_argument(
        "--thinking-path",
        type=str,
        required=True,
        help="Path to the file containing the model's chain of thought outputs.",
    )
    parser.add_argument(
        "--final-response-path",
        type=str,
        required=True,
        help="Path to the file containing the model's final responses.",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Whether to print the evaluation statistics.",
    )


    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_leaks(
        thinking_path=args.thinking_path,
        final_response_path=args.final_response_path,
        print_stats=args.print_stats,
    )

if __name__ == "__main__":
    main()
