import os
import json
from transformers import BertTokenizer, BertModel
import copy

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print(len(tokenizer))

