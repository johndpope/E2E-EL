import os
import json
import numpy as np
from sklearn import preprocessing
import time

stop_words = {"i":1, "me":2, "my":3, "myself":4, "we":5, "our":6, "ours":7, "ourselves":8, "you":9, "your":10, "yours":11, "yourself":12,\
             "yourselves":13, "he":14, "him":15, "his":16, "himself":17, "she":18, "her":19, "hers":20, "herself":21, "it":22, "its":23, "itself":24,\
             "they":25, "them":26, "their":27, "theirs":28, "themselves":29, "what":30, "which":31, "who":32, "whom":33, "this":34, "that":35,\
             "these":36, "those":37, "am":38, "is":39, "are":40, "was":41, "were":42, "be":43, "been":44, "being":45, "have":46, "has":47, "had":48, "having":49,\
             "do":50, "does":51, "did":52, "doing":53, "a":54, "an":55, "the":56, "and":57, "but":58, "if":59, "or":60, "because":61, "as":62, "until":63,\
             "while":64, "of":65, "at":66, "by":67, "for":68, "with":69, "about":70, "against":71, "between":72, "into":73, "through":74, "during":75,\
             "before":76, "after":77, "above":78, "below":79, "to":80, "from":81, "up":82, "down":83, "in":84, "out":85, "on":86, "off":87, "over":88,\
             "under":89, "again":90, "further":91, "then":92, "once":93, "here":94, "there":95, "when":96, "where":97, "why":98, "how":99, "all":100,\
             "any":101, "both":102, "each":103, "few":104, "more":105, "most":106, "other":107, "some":108, "such":109, "no":110, "nor":111, "not":112, "only":113,\
             "own":114, "same":115, "so":116, "than":117, "too":118, "very":119, "s":120, "t":121, "can":122, "will":123, "just":124, "don":125, "should":126, "now":127}

data_dir = 'data/NCBI_Disease/raw_data'
train_data_file = os.path.join(data_dir, 'NCBItrainset_corpus.txt')
test_data_file = os.path.join(data_dir, 'NCBItestset_corpus.txt')
dev_data_file = os.path.join(data_dir, 'NCBIdevelopset_corpus.txt')
entity_file = os.path.join(data_dir, 'entities.txt')

# Entity dictionary (i.e. all possible candidates)
entities = {}
entity_to_index = {}
index_to_entity = {}
with open(entity_file) as f:
    for line in f:
        cols = line.strip().split('\t')
        if cols[0].startswith('MESH'):
            entity_id = cols[0].split(':')[1].strip()       #Training set contains only Dxxx or Cxxx for MESH:Dxxx or MESH:Cxxx
        else:
            entity_id = cols[0].strip()                     #For OMIM, OMIM: is retained in the training set
        entity_text = cols[1]
        if entity_id not in entities:
            entities[entity_id] = entity_text
            entity_index = len(index_to_entity)
            entity_to_index[entity_id] = entity_index
            index_to_entity[entity_index ] = entity_id
print("Total number of unique concepts mentioned ", len(entities))

#print(entities)

# collect the mentions and positive candidates per mention
mentions = {}
with open(train_data_file) as f:
    for line in f:
        cols = line.strip().split('\t')
        if len(cols) == 6:
            document_id = cols[0]
            if document_id not in mentions:
                mentions[document_id] = {}
            mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
            mentions[document_id][mention_id] = {}
            mentions[document_id][mention_id]['text'] = cols[3]
            mentions[document_id][mention_id]['start_index'] = int(cols[1])
            mentions[document_id][mention_id]['end_index'] = int(cols[2])
            if 'st21pv' in data_dir:
                positive_candidate_id = cols[5][5:]
            else:
                if '+' in cols[5]:                        #some concepts are combination of two concepts and hence two ids are concatenated by + or |
                    positive_candidate_id = cols[5].split('+')[0].strip()
                elif '|' in cols[5]:
                    positive_candidate_id = cols[5].split('|')[0].strip()
                else:
                    positive_candidate_id = cols[5].strip()
            #print(positive_candidate_id)
            candidate_text = entities[positive_candidate_id]
            mentions[document_id][mention_id]['positive_candidate'] = {'candidate_id': positive_candidate_id, 'text': candidate_text}

with open(dev_data_file) as f:
    for line in f:
        cols = line.strip().split('\t')
        if len(cols) == 6:
            document_id = cols[0]
            if document_id not in mentions:
                mentions[document_id] = {}
            mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
            mentions[document_id][mention_id] = {}
            mentions[document_id][mention_id]['text'] = cols[3]
            mentions[document_id][mention_id]['start_index'] = int(cols[1])
            mentions[document_id][mention_id]['end_index'] = int(cols[2])
            if 'st21pv' in data_dir:
                positive_candidate_id = cols[5][5:]
            else:
                if '+' in cols[5]:                        #some concepts are combination of two concepts and hence two ids are concatenated by + or |
                    positive_candidate_id = cols[5].split('+')[0].strip()
                elif '|' in cols[5]:
                    positive_candidate_id = cols[5].split('|')[0].strip()
                else:
                    positive_candidate_id = cols[5].strip()
            candidate_text = entities[positive_candidate_id]
            mentions[document_id][mention_id]['positive_candidate'] = {'candidate_id': positive_candidate_id, 'text': candidate_text}

with open(test_data_file) as f:
    for line in f:
        cols = line.strip().split('\t')
        if len(cols) == 6:
            document_id = cols[0]
            if document_id not in mentions:
                mentions[document_id] = {}
            mention_id = document_id + "_" + str(len(mentions[document_id]) + 1)
            mentions[document_id][mention_id] = {}
            mentions[document_id][mention_id]['text'] = cols[3]
            mentions[document_id][mention_id]['start_index'] = int(cols[1])
            mentions[document_id][mention_id]['end_index'] = int(cols[2])
            if 'st21pv' in data_dir:
                positive_candidate_id = cols[5][5:]
            else:
                if '+' in cols[5]:                        #some concepts are combination of two concepts and hence two ids are concatenated by + or |
                    positive_candidate_id = cols[5].split('+')[0].strip()
                elif '|' in cols[5]:
                    positive_candidate_id = cols[5].split('|')[0].strip()
                else:
                    positive_candidate_id = cols[5].strip()
            candidate_text = entities[positive_candidate_id]
            mentions[document_id][mention_id]['positive_candidate'] = {'candidate_id': positive_candidate_id, 'text': candidate_text}

# print(mentions)
print("Total number of mentions ", len(mentions))

# Build the text corpus for tf-idf
# Lower case the strings
text_corpus = []
for e in entities:
    text_corpus.append(entities[e].lower())

for d in mentions:
    for m in mentions[d]:
        text_corpus.append(mentions[d][m]['text'].lower())
print(len(text_corpus))

def getNgramsCount(sentences, n):
    ngrams_count = {}
    for sentence in sentences:
        word_tokens = sentence.split(' ')
        filtered_sentence = [w for w in word_tokens if w not in stop_words] # Remove stopwords
        sentence = " ".join(filtered_sentence)
        for _n in range(1,n+1):
            for pos in range(1, len(sentence)-_n):
                ngram = sentence[pos:pos+_n]
                if ngram in ngrams_count:
                    ngrams_count[ngram] += 1
                else:
                    ngrams_count[ngram] = 1
    return ngrams_count

ngrams = getNgramsCount(text_corpus, 5)
print("Number of unique char n-grams", len(ngrams))
# Keep the top 100,000 most frequent features
sorted_ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
sorted_ngrams = sorted_ngrams[:100000]

# Indexing the character n-grams
ngrams_to_index = {}
index_to_ngram = {}
for ngram, _ in sorted_ngrams:
    if ngram not in ngrams_to_index:
        ngram_index = len(ngrams_to_index)
        ngrams_to_index[ngram] = ngram_index
        index_to_ngram[ngram_index] = ngram

# Calculate IDF
def getNgramsDF(docs, ngrams_to_index, n):
    ngrams_df = np.zeros((len(ngrams_to_index), len(docs)), dtype=np.int16)
    for doc_idx, doc in enumerate(docs):
        word_tokens = doc.split(' ') # Remove stopwords
        filtered_doc = [w for w in word_tokens if w not in stop_words]
        doc = " ".join(filtered_doc)
        for _n in range(1, n+1):
            for pos in range(1, len(doc)-_n):
                ngram = doc[pos:pos+_n]
                if ngram in ngrams_to_index:
                    ngram_index = ngrams_to_index[ngram]
                    ngrams_df[ngram_index, doc_idx] = 1
    return ngrams_df

ngrams_df = getNgramsDF(text_corpus, ngrams_to_index, 5)
ngram_df = np.sum(ngrams_df, axis=1)
N = len(text_corpus)
ngram_idf = np.log(N / ngram_df)


# Candidate vectors
candidate_vectors = []
for e in entities:
    e_text = entities[e]
    e_ngrams = getNgramsCount([e_text], 5)
    v_e = np.zeros(100000)
    for ngram in e_ngrams:
        if ngram in ngrams_to_index:
            ngram_index = ngrams_to_index[ngram]
            tf = e_ngrams[ngram]
            idf = ngram_idf[ngram_index]
            v_e[ngram_index] = tf * idf
    candidate_vectors.append(v_e)

# L2-normalize candidate vectors
candidate_vectors = np.array(candidate_vectors)
candidate_vectors = preprocessing.normalize(candidate_vectors, norm='l2', axis=1)

# Mention vectors and similarity scoring
for d in mentions: # key: document_id
    for m in mentions[d]: # key: mention_id
        m_text = mentions[d][m]['text']
        print("m ", m_text)
        m_ngrams = getNgramsCount([m_text], 5)
        v_m = np.zeros(100000)
        for ngram in m_ngrams:
            if ngram in ngrams_to_index:
                ngram_index = ngrams_to_index[ngram]
                tf = m_ngrams[ngram]
                idf = ngram_idf[ngram_index]
                v_m[ngram_index] = tf * idf
        v_m = preprocessing.normalize(v_m.reshape(1, -1), norm='l2', axis=1)

        similarity_scores = np.matmul(v_m, np.transpose(candidate_vectors)).reshape(-1)
        sorted_candidate_index = np.flip(np.argsort(similarity_scores), axis=0).tolist()

        # Append negative candidates
        mentions[d][m]['all_candidates'] = []
        all_count = 0
        print("+ ", mentions[d][m]['positive_candidate']['text'])
        for candidate_index in sorted_candidate_index:
            candidate_id = index_to_entity[candidate_index]
            candidate_text = entities[candidate_id]
            mentions[d][m]['all_candidates'].append({'candidate_id': candidate_id, 'text': candidate_text})
            print("- ", candidate_text)
            all_count += 1
            if all_count == 10:
                break
        print("---------------------")

save_dir = 'data/NCBI_Disease'
with open(os.path.join(save_dir, 'mentions_candidates_tfidf.json'), 'w+') as f:
    json.dump(mentions, f)
