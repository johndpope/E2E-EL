import os
import json
from transformers import BertTokenizer, BertModel
import re
import pdb

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# print(len(tokenizer))

raw_data_dir = './data/aida-yago2-dataset'
regex = re.compile(r'\(\d+.*\)')

documents = []
mentions = {}
words = []
mention_start_indices = []
mention_end_indices = []
mention_texts = []
label_candidate_ids = []

with open(os.path.join(raw_data_dir, 'AIDA-YAGO2-dataset.tsv'), encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith("-DOCSTART-"):
            document_id = re.findall(regex, line)
            document_id = document_id[0]
            print(document_id)
            segment_id = 0
        else:
            line = line.split("\t")
            if len(line) == 1:
                if len(line[0]) == 0:
                    print("End of sentence")
                    if len(mention_texts) == 0:
                        continue
                    new_document_id = document_id + '_' + str(segment_id)
                    documents.append({"document_id": new_document_id, "text": " ".join(words)})
                    if new_document_id not in mentions:
                        mentions[new_document_id] = []

                    for i, m in enumerate(mention_texts):
                        mentions[new_document_id].append(
                            {
                                "mention_id": new_document_id + '_' + str(i),
                                "start_index": mention_start_indices[i],
                                "end_index": mention_end_indices[i],
                                "text": mention_texts[i],
                                "type": "N/A",
                                "content_document_id": new_document_id,
                                "label_candidate_id": label_candidate_ids[i],
                                "tfidf_candidates": []
                            }
                        )
                    segment_id += 1
                    # Reset
                    words = []
                    mention_start_indices = []
                    mention_end_indices = []
                    mention_texts = []
                    label_candidate_ids = []
                    # pdb.set_trace()
                else:
                    words.append(line[0])
            else:
                if line[-1] == "--NME--":
                    words.append(line[0])
                    continue
                if line[1] == 'B':
                    if len(words) == 0:
                        mention_start_indices.append(len(" ".join(words)))
                    else:
                        mention_start_indices.append(len(" ".join(words)) + 1) # Why +1? add a space
                    words.append(line[0])
                    mention_end_indices.append(len(" ".join(words)))
                    mention_texts.append(line[2])
                    label_candidate_ids.append(line[5])
                elif line[1] == 'I':
                    words.append(line[0])
                    mention_end_indices[len(mention_end_indices) - 1] = len(" ".join(words))


# Write to files
save_dir = './data/MM_full_CUI/collective_el_data_2'
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/train/mentions')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/train/mentions'))
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/train/documents')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/train/documents'))
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/test/mentions')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/test/mentions'))
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/test/documents')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/test/documents'))
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/dev/mentions')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/dev/mentions'))
if not os.path.exists(os.path.join(raw_data_dir, 'collective_el_data_2/dev/documents')):
    os.makedirs(os.path.join(raw_data_dir, 'collective_el_data_2/dev/documents'))

documents_train = []
mentions_train = {}
documents_test = []
mentions_test = {}
documents_dev = []
mentions_dev = {}

for doc in documents:
    document_id = doc["document_id"]
    if 'testa' in document_id:
        documents_dev.append(doc)
    elif 'testb' in document_id:
        documents_test.append(doc)
    else:
        documents_train.append(doc)

for document_id in mentions:
    if 'testa' in document_id:
        mentions_train[document_id] = mentions[document_id]
    elif 'testb' in document_id:
        mentions_test[document_id] = mentions[document_id]
    else:
        mentions_train[document_id] = mentions[document_id]

save_dir = "./data/aida-yago2-dataset/collective_el_data_2"

with open(os.path.join(save_dir, 'train/documents/documents.json'), 'w+') as f:
    for doc in documents_test:
        dict_to_write = doc
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'train/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions_test:
        dict_to_write = json.dumps(mentions_test[document_id])
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'dev/documents/documents.json'), 'w+') as f:
    for doc in documents_test:
        dict_to_write = doc
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'dev/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions_test:
        dict_to_write = json.dumps(mentions_test[document_id])
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'test/documents/documents.json'), 'w+') as f:
    for doc in documents_test:
        dict_to_write = doc
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'test/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions_test:
        dict_to_write = json.dumps(mentions_test[document_id])
        f.write(dict_to_write + '\n')
f.close()













