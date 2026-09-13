from cs336_basics.tokenizer import Tokenizer
import argparse

def parse_args():
    parser=argparse.ArgumentParser(
        description="Train BPE tokenizer"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/raw/TinyStoriesV2-GPT4-train.txt",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000
    )

    parser.add_argument(
            "--special_tokens",
            nargs="+",
            default=["<|endoftext|>"],
        )

    parser.add_argument(
            "--vocab_output_path",
            type=str,
            default="data/tokenizer_config_data/vocab.pkl"
        )

    parser.add_argument(
        "--merges_output_path",
        type=str,
        default="data/tokenizer_config_data/merges.pkl"
    )

    parser.add_argument(
            "--special_tokens_output_path",
            type=str,
            default="data/tokenizer_config_data/special_tokens.pkl"
        )

    parser.add_argument(
        "--verbose", action="store_true", help="是否打印详细进度与耗时信息"
    )

    return parser.parse_args()

if __name__=="__main__":
    args=parse_args()

    print(f"bpe training input load path: {args.input_path}")
    print(f"vocab size: {args.vocab_size}")
    print(f"special tokens: {args.special_tokens}")
    print(f"vocab saved in: {args.vocab_output_path}")
    print(f"merges saved in: {args.merges_output_path}")
    vocab,merges=Tokenizer.train(args.input_path,args.vocab_size,args.special_tokens)
    Tokenizer.to_files(args.vocab_output_path,args.merges_output_path,args.special_tokens_output_path,vocab,merges,args.special_tokens)




