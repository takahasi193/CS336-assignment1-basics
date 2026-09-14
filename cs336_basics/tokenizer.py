import os
import regex
from collections import Counter
import pickle
from typing import Iterator

class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.merges=merges
        if special_tokens:
            vocab_reverse={byte_token:token_id for token_id,byte_token in vocab.items()}
            for special_token in set(special_tokens):
                byte_special_token= special_token.encode('utf-8')
                if byte_special_token not in vocab_reverse:
                    vocab[len(vocab)]=byte_special_token
        self.vocab=vocab
        self.vocab_reverse={byte_token:token_id for token_id,byte_token in self.vocab.items()}
        self.pair2id={pair:i for i,pair in enumerate(merges)}
        self.special_tokens=special_tokens
        self.eos_token=self.vocab_reverse.get("<|endoftext|>".encode("utf-8"),None)

    
    def encode(self,text:str)->list[int]:
        if self.special_tokens:
            sorted_special_tokens=sorted(self.special_tokens,key=len,reverse=True)
            sp_tokens_deal_pattern='('+'|'.join(map(regex.escape,sorted_special_tokens))+')'
            document=regex.split(sp_tokens_deal_pattern,text)
        else:
            document=[text]

        PAT=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenize_pattern=regex.compile(PAT)

        ids=[]
        for doc in document:

            # 如果doc为空，直接跳过
            if not doc:
                continue

            # 如果doc本身就是一个词，直接映射
            byte_doc=doc.encode('utf-8')
            if self.special_tokens and doc in self.special_tokens and byte_doc in self.vocab_reverse:
                ids.append(self.vocab_reverse[byte_doc])
                continue

            # 如果doc包含多个词
            for match in pre_tokenize_pattern.finditer(doc):
                substr=match.group().encode('utf-8')
                byte_substr=[bytes([i]) for i in substr]
                while(len(byte_substr)>=2):
                    new_byte_substr=[]
                    pairs=[(byte_substr[i],byte_substr[i+1]) for i in range(len(byte_substr)-1)]
                    vaild_pairs=[pair for pair in pairs if pair in self.pair2id]
                    if not vaild_pairs:
                        break
                    first_merge_pair=min(vaild_pairs,key=lambda p:self.pair2id[p])
                    index=0
                    while(index<len(byte_substr)):
                        if index<len(byte_substr)-1 and byte_substr[index]==first_merge_pair[0] and byte_substr[index+1]==first_merge_pair[1]:
                            new_byte_substr.append(first_merge_pair[0]+first_merge_pair[1])
                            index+=2

                        else:
                            new_byte_substr.append(byte_substr[index])
                            index+=1
                    byte_substr=new_byte_substr

                ids.extend(self.vocab_reverse[byte] for byte in byte_substr)

        return ids


        

    def decode(self,ids)->str:
       return b"".join(self.vocab[token_id] for token_id in ids).decode('utf-8',errors='replace')

    def encode_iterable(self,iterator:Iterator[str])->Iterator[int]:
        for line in iterator:
            for token_id in self.encode(line):
                yield token_id

            

    @classmethod
    def merge_substr(cls,str_tuple,max_pair,new_token_id)->tuple[int,...]:
        i=0
        new_str_tuple=[]
        while i<len(str_tuple):
            if i<len(str_tuple)-1 and str_tuple[i]==max_pair[0] and str_tuple[i+1]==max_pair[1]:
                new_str_tuple.append(new_token_id)
                i+=2

            else:
                new_str_tuple.append(str_tuple[i])
                i+=1

        return tuple(new_str_tuple)
        

    @classmethod
    def train(cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs):
        with open(input_path,'rb') as f:
            text=f.read().decode('utf-8')
        if special_tokens:
            sorted_special_tokens=sorted(special_tokens,key=len,reverse=True)
            sp_tokens_deal_pattern='|'.join(map(regex.escape,sorted_special_tokens))
            document=regex.split(sp_tokens_deal_pattern,text)
        else:
            document=[text]

        PAT=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenize_pattern=regex.compile(PAT)
        
        substr_counter:dict[tuple[int,...],int]=Counter()
        for doc in document:
            if not doc:
                continue
            for match in pre_tokenize_pattern.finditer(doc):
                substr=match.group()
                if substr:
                    str_tuple=tuple(substr.encode('utf-8'))
                    substr_counter[str_tuple]+=1

        vocab={token:bytes([token]) for token in range(256)}
        merges=[]

        if special_tokens:
            for special_token in special_tokens:
                vocab[len(vocab)]=special_token.encode('utf-8')

        merge_iters=vocab_size-len(vocab)

        # 合成循环
        for _ in range(merge_iters):
            pair_dict={}
            # 对于每个切分后的子串
            for substr,count in substr_counter.items():
                # 对于每对字符
                for i in range(len(substr)-1):
                    pair=(substr[i],substr[i+1])
                    pair_dict[pair]=pair_dict.get(pair,0)+count
            # 如果没有合成字典，则已经没有可以合成的词了，break
            if not pair_dict:
                break

            max_pair=max(pair_dict,key=lambda p:(pair_dict[p],vocab[p[0]],vocab[p[1]]))
            new_token_id=len(vocab)
            new_substr_counter={}

            for substr,count in substr_counter.items():
                if max_pair[0] not in substr or max_pair[1] not in substr:
                    new_substr_counter[substr]=new_substr_counter.get(substr,0)+count    
                    continue
                
                new_substr=Tokenizer.merge_substr(substr,max_pair,new_token_id)
                new_substr_counter[new_substr]=new_substr_counter.get(new_substr,0)+count
                    

            vocab[new_token_id]=vocab[max_pair[0]]+vocab[max_pair[1]]
            substr_counter=new_substr_counter
            merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))

        return vocab,merges

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath,special_tokens_filepath=None):
        with open(vocab_filepath,'rb') as f:
            vocab=pickle.load(f)
        with open(merges_filepath,'rb') as f:
            merges=pickle.load(f)
        if special_tokens_filepath is not None:
            with open(special_tokens_filepath,'rb') as f:
                special_tokens=pickle.load(f)
        else:
            special_tokens=None
        return cls(vocab,merges,special_tokens)

    @classmethod
    def to_files(cls, vocab_filepath, merges_filepath,special_tokens_filepath,vocab,merges,special_tokens):
            with open(vocab_filepath,'wb') as f:
                pickle.dump(vocab,f)
            with open(merges_filepath,'wb') as f:
                pickle.dump(merges,f)
            with open(special_tokens_filepath,'wb') as f:
                pickle.dump(special_tokens,f)
            print(f"vocab 已保存至 {vocab_filepath}")
            print(f"merges 已保存至 {merges_filepath}")
            print(f"special_tokens 已保存至 {special_tokens_filepath}")





     
