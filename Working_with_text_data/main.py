from .tokenizers.tokenizer_v1 import SimpleTokenizerV1
from .tokenizers.tokenizer_v2 import SimpleTokenizerV2
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
    vocab = create_vocab(open("Working_with_text_data/the-verdict.txt").read())

    tokenizer = SimpleTokenizerV2(vocab)

    sample_text1 = "Hello, do you like tea?"
    sample_text2 = "In the sunlit terraces of the palace."
    sample_text = "<|endoftext|> ".join((sample_text1, sample_text2))

    ids = tokenizer.encode(sample_text)

    print(ids)

    decoded_text = tokenizer.decode(ids)

    print(decoded_text)