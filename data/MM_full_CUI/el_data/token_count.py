import transformers
from transformers import BertTokenizer, BertModel

import time
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

max_len = 0
n_ctx = 512
num_docs = 0
num_longer_docs = 0
with open('./dev/documents/documents.json') as f:
    for line in f:
        doc = line.strip()
        doc = json.loads(doc)
        text = doc["text"]

        tokens = tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        num_docs += 1

        if len(input_ids) < n_ctx:
            input_ids = input_ids + [tokenizer.pad_token_id] * (n_ctx - len(input_ids))
        else:
            num_longer_docs += 1

        #output = model(input_ids)



        #print(tokens)
        print(len(input_ids))

        if len(tokens) > max_len:
            max_len = len(tokens)

print(max_len)
print(num_docs)
print(num_longer_docs)

