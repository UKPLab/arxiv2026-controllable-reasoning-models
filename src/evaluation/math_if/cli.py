import argparse
from .evaluation import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for MathIF.")

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the benchmark",
    )
    parser.add_argument(
        "--thinking-path",
        type=str,
        required=True,
        help="Path to the file containing the model's thinking responses.",
    )
    parser.add_argument(
        "--final-ans-path",
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
    evaluate(
        data_path=args.data_path,
        thinking_path=args.thinking_path,
        final_ans_path=args.final_ans_path,
        print_stats=args.print_stats,
    )

if __name__ == "__main__":
    main()
