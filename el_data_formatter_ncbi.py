import os
import json

#with open('./data/MM_full_CUI/candidates_BM25.json') as f:
#    candidates = json.load(f)

import re
regex = re.compile('^\d+\|[a|t]\|')

with open('./data/NCBI_Disease/mentions_candidates_tfidf.json') as f:
    mentions_candidates_tfidf = json.load(f)

documents = {}
mentions = {}
data_dir = './data/NCBI_Disease/raw_data'
save_dir = './data/NCBI_Disease/el_data'
with open(os.path.join(data_dir, 'NCBItestset_corpus.txt')) as f:
    for line in f:
        line = line.strip()
        if regex.match(line):
            match_span = regex.match(line).span()
            start_span_idx = match_span[0]
            end_span_idx = match_span[1]
            document_id = line[start_span_idx: end_span_idx].split("|")[0]
            text = line[end_span_idx:]
            if document_id not in documents:
                documents[document_id] = text
            else:
                documents[document_id] = documents[document_id] + ' ' + text
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
                candidate_id = cols[5].strip()

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
                continue

with open(os.path.join(save_dir, 'test/documents/documents.json'), 'w+') as f:
    for document_id in documents:
        dict_to_write = {"document_id": document_id, "text": documents[document_id]}
        dict_to_write = json.dumps(dict_to_write)
        f.write(dict_to_write + '\n')
f.close()

with open(os.path.join(save_dir, 'test/mentions/mentions.json'), 'w+') as f:
    for document_id in mentions:
        for m in mentions[document_id]:
            dict_to_write = json.dumps(m)
            f.write(dict_to_write + '\n')
f.close()