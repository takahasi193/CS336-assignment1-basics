from cs336_basics.tokenizer import Tokenizer
import argparse
import time

def parse_args():
    parser=argparse.ArgumentParser(
        description="测试tokenizer编码速度"
    )
    parser.add_argument(
                    "--vocab_load_path",
                    type=str,
                    default="data/tokenizer_config_data/vocab.pkl"
                )
    
    parser.add_argument(
            "--merges_load_path",
            type=str,
            default="data/tokenizer_config_data/merges.pkl"
        )

    parser.add_argument(
                "--special_tokens_load_path",
                type=str,
                default="data/tokenizer_config_data/special_tokens.pkl"
            )

    parser.add_argument(
        "--test_raw_data_load_path",
        type=str,
        default="data/raw/TinyStoriesV2-GPT4-valid.txt"
    )

    return parser.parse_args()

def main():
    args=parse_args()
    tokenizer=Tokenizer.from_files(args.vocab_load_path,args.merges_load_path,args.special_tokens_load_path)
    with open(args.test_raw_data_load_path,"r",encoding="utf-8") as f:
        test_text=f.read()
    print("开始测试")
    start_time=time.time()
    tokenizer.encode(test_text)
    comsume_time=time.time()-start_time
    print(f"消耗时间: {comsume_time}")

if __name__=="__main__":
    main()