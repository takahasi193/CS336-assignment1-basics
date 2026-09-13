import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser=argparse.ArgumentParser(
        description="Create tokenizer data"
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
        "--train_raw_data_load_path",
        type=str,
        default="data/raw/TinyStoriesV2-GPT4-train.txt"
    )

    parser.add_argument(
            "--valid_raw_data_load_path",
            type=str,
            default="data/raw/TinyStoriesV2-GPT4-valid.txt"
        )

    parser.add_argument(
            "--train_tokenized_data_save_path",
            type=str,
            default="data/tokenized/TinyStoriesV2-GPT4-train.npy"
        )
    parser.add_argument(
            "--valid_tokenized_data_save_path",
            type=str,
            default="data/tokenized/TinyStoriesV2-GPT4-valid.npy"
    )
    return parser.parse_args()


if __name__=="__main__":
    args=parse_args()

    # 加载分词器
    tokenizer=Tokenizer.from_files(args.vocab_load_path,args.merges_load_path,args.special_tokens_load_path)

    # 加载和编码数据
    with open(args.train_raw_data_load_path,"r",encoding='utf-8') as f:
        train_tokenized_data=tokenizer.encode_iterable(f)
        train_tokenized_data=list(train_tokenized_data)
        train_tokenized_data=np.array(train_tokenized_data,dtype=np.uint16)
    with open(args.valid_raw_data_load_path,"r",encoding='utf-8') as f:
        valid_tokenized_data=tokenizer.encode_iterable(f)
        valid_tokenized_data=list(valid_tokenized_data)
        valid_tokenized_data=np.array(valid_tokenized_data,dtype=np.uint16)

    # 保存数据
    np.save(args.train_tokenized_data_save_path,train_tokenized_data)
    np.save(args.valid_tokenized_data_save_path,valid_tokenized_data)

    print(f"数据已保存至 {args.train_tokenized_data_save_path} 和 {args.valid_tokenized_data_save_path}")
    
    




