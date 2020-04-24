import os
import json
import nltk
import math
import random
import torch
from torch.utils.data import DataLoader, Dataset


def load_data(data_dir, mode):
    entity_path = '../data/MM_full_CUI/raw_data/entities.txt'
    entities = {}
    with open(entity_path, encoding='utf-8') as f:
        for line in f:
            e, type, text = line.strip().split('\t')
            entities[e] = {"text": text.lower(),
                           "type": type}

    file_path = os.path.join(data_dir, mode, 'documents/documents.json')
    docs = {}
    with open(file_path, encoding='utf-8') as f:
        print("documents dataset is loading......")
        for line in f:
            fields = json.loads(line)
            docs[fields["document_id"]] = {"text": fields["text"].lower()}
        print("documents dataset is done :)")

    file_path = os.path.join(data_dir, mode, 'mentions/mentions.json')
    samples = {}
    with open(file_path, encoding='utf-8') as f:
        print("mentions {} dataset is loading......".format(mode))
        for line in f:
            fields = json.loads(line)
            samples[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        print("mentions {} dataset is done :)".format(mode))

    return docs, samples, entities


class DataUtils:
    def __init__(self):
        self.vocab = dict()
        self.word2idx = dict()
        self.idx2word = dict()

        self.glove_embeddings = dict()

        self.char_vocab = dict()
        self.num_char = 0
        self.char2idx = dict()
        self.idx2char = dict()

        self.types = dict()
        self.num_types = 0
        self.type2id = dict()
        self.id2type = dict()

    def build_vocabulary(self, data_dir):
        # Add pad token
        self.vocab['<pad>'] = 1
        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>'
        # Add UNK token
        self.vocab['<unk>'] = 1
        self.word2idx['<unk>'] = 1
        self.idx2word[1] = '<unk>'

        print(" **** Building word and type vocabulary ...")
        entity_path = '../data/MM_full_CUI/raw_data/entities.txt'
        with open(entity_path, encoding='utf-8') as f:
            print("Processing entities ...")
            for line in f:
                _, type, text = line.strip().split('\t')
                text = text.lower()
                words = nltk.word_tokenize(text)
                for w in words:
                    if w not in self.vocab:
                        idx = len(self.vocab)
                        self.word2idx[w] = idx
                        self.idx2word[idx] = w
                        self.vocab[w] = 1
                    else:
                        self.vocab[w] += 1

                if type not in self.types:
                    type_idx = len(self.types)
                    self.type2id[type] = type_idx
                    self.id2type[type_idx] = type
                    self.types[type] = 1
                    self.num_types += 1

        modes = ['train', 'dev', 'test']
        for mode in modes:
            file_path = os.path.join(data_dir, mode, 'documents/documents.json')
            with open(file_path, encoding='utf-8') as f:
                print("Processing {} documents ...".format(mode))
                for line in f:
                    fields = json.loads(line)
                    text = fields["text"].lower()
                    words = nltk.word_tokenize(text)
                    for w in words:
                        if w not in self.vocab:
                            idx = len(self.vocab)
                            self.word2idx[w] = idx
                            self.idx2word[idx] = w
                            self.vocab[w] = 1
                        else:
                            self.vocab[w] += 1

    def build_char_vocabulary(self, data_dir):
        print(" **** Building character vocabulary ...")
        for w in self.vocab:
            for c in w:
                if c not in self.char_vocab:
                    idx = len(self.char_vocab)
                    self.char2idx[c] = idx
                    self.idx2char[idx] = c
                    self.char_vocab[c] = 1
                    self.num_char += 1
                else:
                    self.char_vocab[c] += 1

    def set_glove_embeddings(self, glove_embedding_path):
        print(" **** Loading GloVe embeddings ...")
        with open(glove_embedding_path) as f:
            self.glove_embeddings = json.load(f)

    def get_glove_embeddings(self, w):
        if w in self.glove_embeddings:
            return [self.glove_embeddings[w]]
        else:
            return [[0.0] * 300]

    def convert_index_to_word(self, idx):
        return self.idx2word[idx]

    def convert_word_to_index(self, w):
        if w in self.word2idx:
            return self.word2idx[w]
        else:
            return self.word2idx['<unk>']

    def convert_type_to_id(self, type):
        return self.type2id[type]

    def convert_word_to_char_idx(self, w):
        char_idx = []
        for c in w:
            if c in self.char2idx:
                char_idx.append(self.char2idx[c])
            else:
                char_idx.append(0)
        return char_idx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MedMentions(Dataset):

    def __init__(self, documents, samples, entities, utils, max_len,
                 max_num_candidates, is_training):
        self.documents = documents
        self.samples = samples
        self.entities = entities
        self.utils = utils
        self.max_num_candidates = max_num_candidates
        self.max_len = max_len
        self.max_len_mention = max_len
        self.max_len_candidate = max_len // 2
        self.is_training = is_training

        self.num_samples_original = len(samples)
        # self.samples = self.cull_samples(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mention_id = list(self.samples.keys())[index]

        # Get mention context
        mention_window, mention_start, mention_end \
            = self.get_mention_window(mention_id, self.samples, self.documents, self.max_len)

        # Tokenize mention context
        mention_token_ids = self.convert_tokens_to_ids(mention_window, self.max_len_mention)

        # Encode mention type
        mention_type_id = self.convert_type_to_id(self.samples[mention_id]["type"])

        # Prepare candidates and target entity
        candidates = self.samples[mention_id]["tfidf_candidates"]
        candidate_ids, target_index = self.prepare_candidates(mention_id, self.samples, candidates)

        encoded_candidates = []
        candidate_types = []
        for i, candidate_id in enumerate(candidate_ids):
            # Tokenize candidate entity name
            entity_text = self.entities[candidate_id]["text"]
            candidate_prefix = self.get_entity_window(entity_text)
            candidate_token_ids = self.convert_tokens_to_ids(candidate_prefix, self.max_len_candidate)
            encoded_candidates.append(candidate_token_ids)

            # Encode candidate entity type
            candidate_type = self.entities[candidate_id]["type"]
            candidate_type_id = self.convert_type_to_id(candidate_type)
            candidate_types.append([candidate_type_id])

        # Convert everything to tensors
        encoded_mention = torch.LongTensor(mention_token_ids)
        encoded_candidates = torch.LongTensor(encoded_candidates)
        target_candidate = torch.LongTensor([target_index])
        candidate_types = torch.LongTensor(candidate_types)
        mention_type = torch.LongTensor([mention_type_id])

        return encoded_mention, encoded_candidates, target_candidate, mention_type, candidate_types

    def prepare_candidates(self, mention_id, samples, candidates):
        xs = candidates[:self.max_num_candidates]
        y = samples[mention_id]["label_candidate_id"]

        if self.is_training:
            # At training time we can include target if not already included.
            if not y in xs:
                xs.append(y)
        # else:
        #     # At test time we assume candidates already include target.
        #     assert y in xs

        xs = [y] + [x for x in xs if x != y]  # Target index always 0
        xs = xs[:self.max_num_candidates]
        y_idx = xs.index(y)
        return xs, y_idx

    def get_window(self, prefix, mention, suffix, max_size):
        if len(mention) >= max_size:
            window = mention[:max_size]
            return window, 0, len(window) - 1

        leftover = max_size - len(mention)
        leftover_half = int(math.ceil(leftover / 2))

        if len(prefix) >= leftover_half:
            prefix_len = leftover_half if len(suffix) >= leftover_half else \
                leftover - len(suffix)
        else:
            prefix_len = len(prefix)

        prefix = prefix[-prefix_len:]  # Truncate head of prefix
        window = prefix + mention + suffix
        window = window[:max_size]  # Truncate tail of suffix

        mention_start_index = len(prefix)
        mention_end_index = len(prefix) + len(mention) - 1

        return window, mention_start_index, mention_end_index

    def get_mention_window(self, mention_id, mentions, docs, max_seq_length):
        max_len_context = max_seq_length  # number of characters
        # Get "enough" context from text.
        content_document_id = mentions[mention_id]['content_document_id']
        context_text = docs[content_document_id]['text']
        start_index = mentions[mention_id]['start_index']
        end_index = mentions[mention_id]['end_index']
        prefix = context_text[max(0, start_index - max_len_context): start_index]
        suffix = context_text[end_index: end_index + max_len_context]
        extracted_mention = context_text[start_index: end_index]

        assert extracted_mention == mentions[mention_id]['text'].lower()

        # Get window under new tokenization.
        return self.get_window(nltk.word_tokenize(prefix),
                          nltk.word_tokenize(extracted_mention),
                          nltk.word_tokenize(suffix),
                          max_len_context)

    def get_entity_window(self, entity_text):
        entity_tokens = nltk.word_tokenize(entity_text)
        if len(entity_tokens) > self.max_len_candidate:
            entity_tokens = entity_tokens[:self.max_len_candidate]
        return entity_tokens

    def convert_tokens_to_ids(self, tokens, max_len):
        token_ids = [self.utils.convert_word_to_index(t) for t in tokens]
        if len(tokens) < max_len:
            padding_len = max_len - len(tokens)
            token_ids = token_ids + [self.utils.convert_word_to_index('<pad>')] * padding_len
        return token_ids

    def convert_type_to_id(self, type):
        return self.utils.convert_type_to_id(type)


    def cull_samples(self, samples):
        self.num_samples_original = len(samples)
        if self.is_training:
            return samples
        else:
            return [mc for mc in samples if
                    mc[0]['label_document_id'] in
                    mc[1]['tfidf_candidates'][:self.max_num_candidates]]


def get_loaders(data_dir, utils, max_len, max_num_candidates, batch_size,
                num_workers):

    train_documents, samples_train, entities = load_data(data_dir, 'train')
    dev_documents, samples_dev, entities = load_data(data_dir, 'dev')
    test_documents, samples_test, entities = load_data(data_dir, 'test')

    # Datasets
    dataset_train = MedMentions(train_documents, samples_train, entities, utils,
                                  max_len, max_num_candidates, True)
    dataset_dev = MedMentions(dev_documents, samples_dev, entities, utils,
                                max_len, max_num_candidates, False)
    dataset_test = MedMentions(test_documents, samples_test, entities, utils,
                                 max_len, max_num_candidates, False)

    # Data Loaders
    loader_train = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    loader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return loader_train, loader_dev, loader_test

# data_dir = '../data/MM_full_CUI/el_data'
# utils = DataUtils()
# utils.build_vocabulary(data_dir)
# max_len = 32
# max_num_candidates = 10
# batch_size = 2
# num_workers = 2
# loader_train, loader_dev, loader_test = get_loaders(data_dir, utils, max_len, max_num_candidates, batch_size,
#                 num_workers)
#
# for idx, batch in enumerate(loader_train):
#     print(batch[0])
#     print(batch[1])
#     print(batch[2])
#     print(batch[3])
#     print(batch[4])
#     break


