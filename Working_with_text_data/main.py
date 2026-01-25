from tokenizers.tokenizer_v1 import SimpleTokenizerV1
from tokenizers.tokenizer_v2 import SimpleTokenizerV2
from datasets.GPTDatasetV1 import create_dataloader_v1
import tiktoken
import pandas as pd
import re

def create_vocab(text):
    
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer, token in enumerate(all_tokens)}
    return vocab

if __name__ == "__main__":
    # vocab = create_vocab(open("the-verdict.txt").read())

    # tokenizer = SimpleTokenizerV2(vocab)
    
    tokenizer = tiktoken.get_encoding('gpt2')

    with open("Working_with_text_data/the-verdict.txt", 'r', encoding='utf-8') as f: 
        raw_text = f.read()

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)