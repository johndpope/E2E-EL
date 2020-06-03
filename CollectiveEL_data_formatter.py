import os
import json
from transformers import BertTokenizer, BertModel
import copy

import pdb

#with open('./data/MM_full_CUI/candidates_BM25.json') as f:
#    candidates = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

import re
regex = re.compile('^\d+\|[a|t]\|')

with open('./data/MM_full_CUI/mentions_candidates_tfidf.json') as f:
    mentions_candidates_tfidf = json.load(f)

documents = {}
mentions = {}
data_dir = './data/MM_full_CUI/raw_data'
save_dir = './data/MM_full_CUI/collective_el_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(data_dir, 'test_corpus.txt')) as f:
    for line in f:
        line = line.strip()
        if regex.match(line):
            match_span = regex.match(line).span()
            start_span_idx = match_span[0]
            end_span_idx = match_span[1]
            document_id = line[start_span_idx: end_span_idx].split("|")[0]
            text = line[end_span_idx:]
            if document_id not in documents:
                documents[document_id] = text # Title is added
            else:
                documents[document_id] = documents[document_id] + ' ' + text # Abstract is added
        else:
            cols = line.strip().split('\t')
            if len(cols) == 6:
                document_id = cols[0]
                if document_id not in mentions:
                    mentions[document_id] = []
                mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
                start_index = cols[1]
                end_index = cols[2]
                mention_text = cols[3]
                mention_type = cols[4] #.split(',')[0]
                candidate_id = cols[5]

                # Positive and negative candidates
                # tfidf_candidates = []
                # pos_candidate = mentions_candidates_tfidf[document_id][mention_id]["positive_candidate"]["candidate_id"]
                # tfidf_candidates.append(pos_candidate)
                # neg_candidates = []
                # for nc in mentions_candidates_tfidf[document_id][mention_id]["negative_candidate"]:
                #     neg_candidates.append(nc["candidate_id"])
                # tfidf_candidates += neg_candidates

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
                                         "tfidf_candidates": tfidf_candidates #list(candidates[candidate_id]["candidates"].keys())
                                         })

            else: # Empty lines
                print(document_id)
                continue


all_documents = copy.deepcopy(documents)

for document_id in all_documents:
    full_doc_text = all_documents[document_id]
    sentences = full_doc_text.strip().split('. ')
    # print(sentences)
    num_sentences = len(sentences)
    # print(num_sentences)

    segment_num = 1
    segment_text = ""
    all_mentions = mentions[document_id]
    # print(len(all_mentions))
    # print("------")
    mentions_within_segment = []
    # Checking whether the number of token exceeds the max context length allowed by BERT
    for s_idx, s in enumerate(sentences):
        # print(s)
        # Add '. ' to every sentence except the last one
        if s_idx < num_sentences - 1:
            s = s + '. '
        # Add the sentence to the existing segment
        # and check whether the tokenized length of that is less that 512
        tokens = tokenizer.tokenize(segment_text + s)
        if len(['[CLS]'] + tokens + ['[SEP]']) <= 512:
            segment_text = segment_text + s
            continue
        else:
            # print("Segmentation needed!!!")
            # Make the seqment a new document
            new_document_id = document_id + '_' + str(segment_num)
            documents[new_document_id] = segment_text
            segment_len = len(segment_text)

            # Update segment number
            segment_num += 1

            # Start a new segment
            segment_text = ""
            # Add the current sentence to the new segment
            segment_text = segment_text + s

            # Reset `mentions_within_segment`
            mentions_within_segment = []

            # Find the mentions that are within this segment
            for k in range(len(all_mentions)):
                # print(k)
                if all_mentions[k]['end_index'] < segment_len:
                    mentions_within_segment.append(all_mentions[k])
                else:
                    break

            if k == len(all_mentions) - 1:
                all_mentions = all_mentions[k+1:]
            else:
                all_mentions = all_mentions[k:]
            # print(all_mentions)
            for mention in all_mentions:
                mention['start_index'] = mention['start_index'] - segment_len
                mention['end_index'] = mention['end_index'] - segment_len

            # Add the mentions within the segment to the `mentions` dictionary
            mentions[new_document_id] = []
            for i, m in enumerate(mentions_within_segment):
                # Change the mention_id and content_document_id
                m['content_document_id'] = new_document_id
                m['mention_id'] = new_document_id + '_' + str(i+1)
                mentions[new_document_id].append(m)
            # pdb.set_trace()
            # print(mentions[new_document_id])

    # For the final segment
    # Make the seqment a new document
    new_document_id = document_id + '_' + str(segment_num)
    documents[new_document_id] = segment_text
    segment_len = len(segment_text)

    # Add the mentions within the segment to the `mentions` dictionary
    mentions[new_document_id] = []
    mentions_within_segment = all_mentions # the remaining mentions in `all_mentions`
    if len(mentions_within_segment ) > 0:
        for k, m in enumerate(mentions_within_segment ):
            # Change the mention_id and content_document_id
            m['content_document_id'] = new_document_id
            m['mention_id'] = new_document_id + '_' + str(k + 1)
            mentions[new_document_id].append(m)
    else:
        if segment_num == 1:
            # If there is only one segment then the `else` part of the above block is bypassed, so ...
            mentions_within_segment = copy.deepcopy(mentions[document_id])
            for k, m in enumerate(mentions_within_segment):
                # Change the mention_id and content_document_id
                m['content_document_id'] = new_document_id
                m['mention_id'] = new_document_id + '_' + str(k + 1)
                mentions[new_document_id].append(m)
        else:
            print("Empty mentions for ", new_document_id)
            mentions[new_document_id] = []

    # assert len(mentions[new_document_id]) == len(mentions[document_id])

    # Delete the original document from both`
    del documents[document_id]
    # print(len(mentions[document_id]))
    del mentions[document_id]

# print("*** Documents ***")
# print(len(documents))
# for d in documents:
#     print(d, documents[d])
#
# print("*** Mentions ***")
# print(len(mentions))
# for d in mentions:
#     print(d, mentions[d])

if not os.path.exists(os.path.join(save_dir, "test/documents")):
    os.makedirs(os.path.join(save_dir, "test/documents"))
with open(os.path.join(save_dir, 'test/documents/documents.json'), 'w+') as f:
    for document_id in documents:
        dict_to_write = {"document_id": document_id, "text": documents[document_id]}
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

if not os.path.exists(os.path.join(save_dir, "test/mentions")):
    os.makedirs(os.path.join(save_dir, "test/mentions"))
with open(os.path.join(save_dir, 'test/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions:
        # for m in mentions[document_id]:
        dict_to_write = json.dumps(mentions[document_id])
        f.write(dict_to_write + '\n')
f.close()