import os
import json
from transformers import BertTokenizer, BertModel
import copy
from time import time

import pdb

#with open('./data/MM_full_CUI/candidates_BM25.json') as f:
#    candidates = json.load(f)

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='./biobert_v1.1_pubmed', do_lower_case=False, cache_dir=None)
print(len(tokenizer))
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print(len(tokenizer))


import re
regex = re.compile('^\d+\|[a|t]\|')

with open('./data/BC5CDR/mentions_candidates_tfidf.json') as f:
    mentions_candidates_tfidf = json.load(f)

documents = {}
mentions = {}
data_dir = './data/BC5CDR/raw_data'
save_dir = './data/BC5CDR/collective_el_data_2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(data_dir, 'train_corpus.txt')) as f:
    for line in f:
        line = line.strip()
        if regex.match(line):
            match_span = regex.match(line).span()
            start_span_idx = match_span[0]
            end_span_idx = match_span[1]
            document_id = line[start_span_idx:end_span_idx].split("|")[0]
            text = line[end_span_idx:]
            if document_id not in documents:
                documents[document_id] = text  # Title is added
            else:
                documents[document_id] = documents[document_id] + ' ' + text  # Abstract is added
            print(document_id)
        else:
            cols = line.strip().split('\t')
            if len(cols) == 6:
                if cols[5] == '-1':
                    continue
                document_id = cols[0]
                if document_id not in mentions:
                    mentions[document_id] = []
                mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
                start_index = cols[1]
                end_index = cols[2]
                if int(start_index) >= int(end_index):
                    print("====>", document_id, start_index, end_index)
                mention_text = cols[3]
                mention_type = cols[4]
                candidate_id = cols[5]

                tfidf_candidates = []
                for c in mentions_candidates_tfidf[document_id][mention_id]["all_candidates"]:
                    tfidf_candidates.append(c["candidate_id"])

                mentions[document_id].append({"mention_id": mention_id,
                                         "start_index": int(start_index),
                                         "end_index": int(end_index),
                                         "text": mention_text,
                                         "type": mention_type,
                                         "content_document_id": document_id,
                                         "label_candidate_id": candidate_id,
                                         "tfidf_candidates": tfidf_candidates})

            else: # Empty lines
                continue
print("Segmentation starts ...")

all_documents = copy.deepcopy(documents)

for document_id in all_documents:
    print("Doc ==>", document_id)
    start_index_new_doc = 0
    segment_id = 0
    segment_text = ""
    num_mentions = len(mentions[document_id])
    cumulative_seg_len = [0]
    max_mention_per_new_doc = 8
    num_mentions_new_doc = 0

    print(num_mentions)
    for i in range(num_mentions):
        end_index_new_doc = mentions[document_id][i]["end_index"]
        tentative_segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
        tokens = tokenizer.tokenize(tentative_segment_text)
        if num_mentions_new_doc != max_mention_per_new_doc and len(['[CLS]'] + tokens + ['[SEP]']) < 256:
            num_mentions_new_doc += 1
            segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
            start_index_new_doc = end_index_new_doc
            # Add the mention to `mentions`
            new_document_id = document_id + '_' + str(segment_id)
            if new_document_id not in mentions:
                mentions[new_document_id] = []
            mention_id = new_document_id + '_' + str(i % max_mention_per_new_doc)
            new_mention = copy.deepcopy(mentions[document_id][i])
            new_mention["mention_id"] = mention_id
            new_mention["content_document_id"] = new_document_id
            new_mention["start_index"] = new_mention["start_index"] - cumulative_seg_len[segment_id]
            new_mention["end_index"] = new_mention["end_index"] - cumulative_seg_len[segment_id]
            mentions[new_document_id].append(new_mention)
            continue
        else:
            # Write the new segment
            new_document_id = document_id + '_' + str(segment_id)
            if new_document_id not in mentions:
                mentions[new_document_id] = []
            documents[new_document_id] = segment_text
            cumulative_seg_len.append(cumulative_seg_len[-1] + len(segment_text))

            # Reset everything
            num_mentions_new_doc = 0
            segment_text = ""

            # Increment segment number
            segment_id += 1

            # Take care of the current mention for which if condition returned False
            num_mentions_new_doc += 1
            segment_text = segment_text + documents[document_id][start_index_new_doc:end_index_new_doc]
            start_index_new_doc = end_index_new_doc

            # Add the mention to `mentions`
            new_document_id = document_id + '_' + str(segment_id)
            if new_document_id not in mentions:
                mentions[new_document_id] = []
            mention_id = new_document_id + '_' + str(i % max_mention_per_new_doc)
            new_mention = copy.deepcopy(mentions[document_id][i])
            new_mention["mention_id"] = mention_id
            new_mention["content_document_id"] = new_document_id
            new_mention["start_index"] = new_mention["start_index"] - cumulative_seg_len[segment_id]
            new_mention["end_index"] = new_mention["end_index"] - cumulative_seg_len[segment_id]
            mentions[new_document_id].append(new_mention)

    # Last few mentions and the remaining text
    segment_text = segment_text + documents[document_id][start_index_new_doc:]
    new_document_id = document_id + '_' + str(segment_id)
    documents[new_document_id] = segment_text
    # cumulative_seg_len += len(segment_text)
    cumulative_seg_len.append(cumulative_seg_len[-1] + len(segment_text))
    print(cumulative_seg_len)

    # Delete the original document from both`
    del documents[document_id]
    # Delete the original mentions from `mentions`
    del mentions[document_id]
    # print("*** Documents ***")
    # for d in documents:
    #     print(d, documents[d])

    # segment_id = 0
    # new_document_id = document_id + '_' + str(segment_id)
    # mentions[new_document_id] = []
    # for i in range(num_mentions):
    #     print(segment_id)
    #     if (i % max_mention_per_new_doc == 0 and i > 0) or cumulative_seg_len[segment_id+1] < mentions[document_id][i]['end_index']:
    #         segment_id += 1
    #         new_document_id = document_id + '_' + str(segment_id)
    #         mentions[new_document_id] = []
    #     # Add the mention to `mentions`
    #     mention_id = new_document_id + '_' + str(i % max_mention_per_new_doc)
    #     new_mention = copy.deepcopy(mentions[document_id][i])
    #     new_mention["mention_id"] = mention_id
    #
    #     new_mention["content_document_id"] = new_document_id
    #     new_mention["start_index"] = new_mention["start_index"] - cumulative_seg_len[segment_id]
    #     new_mention["end_index"] = new_mention["end_index"] - cumulative_seg_len[segment_id]
    #     mentions[new_document_id].append(new_mention)
    #
    # # Delete the original mentions from `mentions`
    # del mentions[document_id]

    # print("*** Mentions ***")
    # for m in mentions:
    #     print(m, mentions[m])


if not os.path.exists(os.path.join(save_dir, "train/documents")):
    os.makedirs(os.path.join(save_dir, "train/documents"))
with open(os.path.join(save_dir, 'train/documents/documents.json'), 'w+') as f:
    for document_id in documents:
        dict_to_write = {"document_id": document_id, "text": documents[document_id]}
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

if not os.path.exists(os.path.join(save_dir, "train/mentions")):
    os.makedirs(os.path.join(save_dir, "train/mentions"))
with open(os.path.join(save_dir, 'train/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions:
        # for m in mentions[document_id]:
        dict_to_write = json.dumps(mentions[document_id])
        f.write(dict_to_write + '\n')
f.close()