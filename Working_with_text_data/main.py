from tokenizers.tokenizer_v1 import SimpleTokenizerV1
from tokenizers.tokenizer_v2 import SimpleTokenizerV2
from datasets.GPTDatasetV1 import create_dataloader_v1
import tiktoken
import pandas as pd
import re
import torch

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

    max_length=4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
    
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    token_embeddings = token_embedding_layer(inputs)

    context_length = max_length
    pos_emedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_emedding_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
