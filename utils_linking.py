import os
import json
import random
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
logger = logging.getLogger(__name__)

# word_piece_ment = 128
# word_piece_ent = 128

# import tokenization
# tokenizer = tokenization.BasicTokenizer(do_lower_case=False)


def get_examples(data_dir, mode):
    entity_path = './data/MM_full_CUI/raw_data/entities.txt'
    entities = {}
    with open(entity_path, encoding='utf-8') as f:
        for line in f:
            e, _, text = line.strip().split('\t')
            entities[e] = text

    file_path = os.path.join(data_dir, mode, 'documents/documents.json')
    docs = {}
    with open(file_path, encoding='utf-8') as f:
        print("documents dataset is loading......")
        for line in f:
            fields = json.loads(line)
            docs[fields["document_id"]] = {"text": fields["text"]}
        print("documents dataset is done :)")

    file_path = os.path.join(data_dir, mode, 'mentions/mentions.json')
    ments = {}
    with open(file_path, encoding='utf-8') as f:
        print("mentions {} dataset is loading......".format(mode))
        for line in f:
            fields = json.loads(line)
            ments[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        print("mentions {} dataset is done :)".format(mode))

    return ments, docs, entities

def get_window(prefix, mention, suffix, max_size):
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


def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 3 // 2 # number of characters
    # Get "enough" context from space-tokenized text.
    content_document_id = mentions[mention_id]['content_document_id']
    # print(content_document_id)
    context_text = docs[content_document_id]['text']
    start_index = mentions[mention_id]['start_index']
    end_index = mentions[mention_id]['end_index']
    prefix = context_text[max(0, start_index - max_len_context): start_index]
    suffix = context_text[end_index: end_index + max_len_context]
    extracted_mention = context_text[start_index: end_index]
    # print(len(context_text))
    # print(start_index)
    # print(end_index)
    # print(extracted_mention)
    # print(mentions[mention_id]['text'])
    assert extracted_mention == mentions[mention_id]['text']

    # Get window under new tokenization.
    return get_window(tokenizer.tokenize(prefix),
                      tokenizer.tokenize(extracted_mention),
                      tokenizer.tokenize(suffix),
                      max_len_context)


def get_entity_window(entity_text, max_entity_len, tokenizer):
    entity_tokens = tokenizer.tokenize(entity_text)
    if len(entity_tokens) > max_entity_len:
        entity_tokens = entity_tokens[:max_entity_len]
    return entity_tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, mention_boundary_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mention_boundary_ids = mention_boundary_ids
        self.label_ids = label_ids

def convert_examples_to_features(
    mentions,
    docs,
    entities,
    max_seq_length,
    tokenizer,
    mode,
):

    features = []
    for (ex_index, mention_id) in enumerate(mentions.keys()):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(mentions))

        mention_window, mention_start_index, mention_end_index = get_mention_window(mention_id,
                                                                            mentions,
                                                                            docs,
                                                                            max_seq_length,
                                                                            tokenizer)

        input_ids = []
        input_mask = []
        segment_ids = []
        mention_boundary_ids = []
        label_ids = []

        # List of candidates
        label_candidate_id = mentions[mention_id]['label_candidate_id']
        candidates = []
        if mode == 'train':
            candidates.append(label_candidate_id) # positive candidate
            for c in mentions[mention_id]["tfidf_candidates"]:
                if c != label_candidate_id and len(candidates) < 10:
                    candidates.append(c)
        else:
            for c in mentions[mention_id]["tfidf_candidates"]:
                candidates.append(c)

        random.shuffle(candidates)
        # Target candidate
        if label_candidate_id in candidates:
            label_id = [candidates.index(label_candidate_id)]
        else:
            label_id = [len(candidates) + 1] # when target candidate not in candidate set

        for c_idx, c in enumerate(candidates):
            entity_text = entities[c]
            max_entity_len = max_seq_length - ((max_seq_length - 3) // 2) # Number of tokens
            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)

            # [CLS] mention with context [SEP] candidate [SEP]
            tokens_pair = [tokenizer.cls_token] + mention_window + [tokenizer.sep_token] + entity_window + [tokenizer.sep_token]
            tokens_pair = tokenizer.convert_tokens_to_ids(tokens_pair)

            # Token types
            num_zeros = len(mention_window) + 2 # [CLS] and first [SEP]
            num_ones = len(entity_window) + 1 # second [SEP]
            token_type_id = [0] * num_zeros + [1] * num_ones

            # mention boundary
            mention_boundary_id = [0] * len(tokens_pair)
            for i in range(mention_start_index, mention_end_index + 1):
                mention_boundary_id[i] = 1

            if len(tokens_pair) > max_seq_length:
                tokens_pair = tokens_pair[:max_seq_length]
                token_type_id = token_type_id[:max_seq_length]
                mention_boundary_id = mention_boundary_id[:max_seq_length]
                tokens_mask = [1]*max_seq_length
            else:
                pair_len = len(tokens_pair)
                pad_len = max_seq_length - pair_len
                tokens_pair = tokens_pair + tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
                token_type_id = token_type_id + tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
                mention_boundary_id = mention_boundary_id + tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
                tokens_mask = [1]*pair_len + [0] * pad_len

            assert len(tokens_pair) == max_seq_length
            assert len(token_type_id) == max_seq_length
            assert len(mention_boundary_id) == max_seq_length
            assert len(tokens_mask) == max_seq_length

            input_ids.append(tokens_pair)
            input_mask.append(tokens_mask)
            segment_ids.append(token_type_id)
            mention_boundary_ids.append(mention_boundary_id)
            label_ids.append(label_id)

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("guid: %s", .guid)
            # logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids[ex_index]]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask[ex_index]]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids[ex_index]]))
            logger.info("mention_boundary_ids: %s", " ".join([str(x) for x in mention_boundary_ids[ex_index]]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids[ex_index]]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          mention_boundary_ids=mention_boundary_ids,
                          label_ids=label_ids,
                          )
        )
    return features
