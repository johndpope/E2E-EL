import os
import json
import random
import math
import logging
logger = logging.getLogger(__name__)
import torch
import faiss

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

def get_window_for_retrieval(prefix, mention, suffix, max_size):
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
    window = prefix + ['[Ms]'] + mention + ['[Me]'] + suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index


def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 3 // 2 # number of characters
    # Get "enough" context from space-tokenized text.
    content_document_id = mentions[mention_id]['content_document_id']
    context_text = docs[content_document_id]['text']
    start_index = mentions[mention_id]['start_index']
    end_index = mentions[mention_id]['end_index']
    prefix = context_text[max(0, start_index - max_len_context): start_index]
    suffix = context_text[end_index: end_index + max_len_context]
    extracted_mention = context_text[start_index: end_index]

    assert extracted_mention == mentions[mention_id]['text']

    # Get window under new tokenization.
    return get_window(tokenizer.tokenize(prefix),
                      tokenizer.tokenize(extracted_mention),
                      tokenizer.tokenize(suffix),
                      max_len_context)

def get_mention_window_for_retrieval(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 3 // 2 # number of characters
    # Get "enough" context from space-tokenized text.
    content_document_id = mentions[mention_id]['content_document_id']
    context_text = docs[content_document_id]['text']
    start_index = mentions[mention_id]['start_index']
    end_index = mentions[mention_id]['end_index']
    prefix = context_text[max(0, start_index - max_len_context): start_index]
    suffix = context_text[end_index: end_index + max_len_context]
    extracted_mention = context_text[start_index: end_index]

    assert extracted_mention == mentions[mention_id]['text']

    # Get window under new tokenization.
    return get_window_for_retrieval(tokenizer.tokenize(prefix),
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
    args,
    retrieval_model=None,
    retrieval_tokenizer=None,
):
    if args.use_dense_candidates:
        # All entities
        all_entities = list(entities.keys())
        all_entity_token_ids = []
        all_entity_token_masks = []

        for c_idx, c in enumerate(all_entities):
            entity_text = entities[c]
            max_entity_len = max_seq_length // 2  # Number of tokens in entity
            entity_window = get_entity_window(entity_text, max_entity_len, retrieval_tokenizer)
            # [CLS] candidate text [SEP]
            candidate_tokens = [retrieval_tokenizer.cls_token] + entity_window + [retrieval_tokenizer.sep_token]
            candidate_tokens = retrieval_tokenizer.convert_tokens_to_ids(candidate_tokens)
            if len(candidate_tokens) > max_entity_len :
                candidate_tokens = candidate_tokens[:max_entity_len]
                candidate_masks = [1] * max_entity_len
            else:
                candidate_len = len(candidate_tokens)
                pad_len = max_entity_len - candidate_len
                candidate_tokens += [retrieval_tokenizer.pad_token_id] * pad_len
                candidate_masks = [1] * candidate_len + [0] * pad_len

            assert len(candidate_tokens) == max_entity_len
            assert len(candidate_masks) == max_entity_len

            all_entity_token_ids.append(candidate_tokens)
            all_entity_token_masks.append(candidate_masks)

        if retrieval_model is None:
            raise ValueError("`retrieval_model` parameter cannot be None")
        logger.info("INFO: Building index of the candidate embeddings ...")
        # Gather all candidate embeddings for hard negative mining
        all_candidate_embeddings = []
        with torch.no_grad():
            # Forward pass through the candidate encoder of the dual encoder
            for i, (entity_tokens, entity_tokens_masks) in enumerate(
                    zip(all_entity_token_ids, all_entity_token_masks)):
                candidate_token_ids = torch.LongTensor([entity_tokens]).to(args.device)
                candidate_token_masks = torch.LongTensor([entity_tokens_masks]).to(args.device)
                candidate_outputs = retrieval_model.bert_candidate.bert(
                    input_ids=candidate_token_ids,
                    attention_mask=candidate_token_masks,
                )
                candidate_embedding = candidate_outputs[1]
                all_candidate_embeddings.append(candidate_embedding)

        all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0)

        # Indexing for faster search (using FAISS)
        d = all_candidate_embeddings.size(1)
        all_candidate_index = faiss.IndexFlatL2(d)  # build the index, d=size of vectors
        # here we assume `all_candidate_embeddings` contains a n-by-d numpy matrix of type float32
        all_candidate_embeddings = all_candidate_embeddings.cpu().detach().numpy()
        all_candidate_index.add(all_candidate_embeddings)

    def get_dense_retrieval_candidates(mention_window_for_retrieval):
        max_mention_len = max_seq_length // 2
        # Prepre input for the dual encoder model i.e., [CLS] mention with context [SEP]
        mention_tokens = [retrieval_tokenizer.cls_token] + mention_window_for_retrieval + [
            retrieval_tokenizer.sep_token]
        mention_tokens = retrieval_tokenizer.convert_tokens_to_ids(mention_tokens)
        if len(mention_tokens) > max_mention_len:
            mention_tokens = mention_tokens[:max_mention_len]
            mention_tokens_mask = [1] * max_mention_len
        else:
            mention_len = len(mention_tokens)
            pad_len = max_mention_len - mention_len
            mention_tokens += [retrieval_tokenizer.pad_token_id] * pad_len
            mention_tokens_mask = [1] * mention_len + [0] * pad_len

        assert len(mention_tokens) == max_mention_len
        assert len(mention_tokens_mask) == max_mention_len

        input_token_ids = torch.LongTensor([mention_tokens]).to(args.device)
        input_token_masks = torch.LongTensor([mention_tokens_mask]).to(args.device)
        # Forward pass through the mention encoder of the dual encoder
        with torch.no_grad():
            mention_outputs = retrieval_model.bert_mention.bert(
                input_ids=input_token_ids,
                attention_mask=input_token_masks,
            )
            mention_embedding = mention_outputs[1]  # 1 X d
            mention_embedding = mention_embedding.cpu().detach().numpy()

        # Perform similarity search
        distance, candidate_indices = all_candidate_index.search(mention_embedding, args.num_candidates)
        candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10

        return candidate_indices

    #~~~~~~~~~~~~~~~~~~~~~~~~~ Convert examples to features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    features = []
    position_of_positive = {}
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
        if args.do_train:
            if args.use_tfidf_candidates:
                candidates.append(label_candidate_id) # positive candidate
                for c in mentions[mention_id]["tfidf_candidates"]:
                    if c != label_candidate_id and len(candidates) < 10:
                        candidates.append(c)

            elif args.use_dense_candidates:
                if retrieval_model is None:
                    raise ValueError("`retrieval_model` parameter cannot be None")
                mention_window_for_retrieval, _, _ = get_mention_window_for_retrieval(mention_id,
                                                                                      mentions,
                                                                                      docs,
                                                                                      max_seq_length,
                                                                                      tokenizer)

                candidates.append(label_candidate_id)  # positive candidate
                # Append the retrieved candidates to the list candidates
                candidate_indices = get_dense_retrieval_candidates(mention_window_for_retrieval)
                for i, c_idx in enumerate(candidate_indices):
                    c = all_entities[c_idx]
                    if c == label_candidate_id:
                        if i not in position_of_positive:
                            position_of_positive[i] = 1
                        else:
                            position_of_positive[i] += 1
                    if c != label_candidate_id and len(candidates) < args.num_candidates:
                        candidates.append(c)

        if args.do_eval:
            if args.use_tfidf_candidates:
                for c in mentions[mention_id]["tfidf_candidates"]:
                    candidates.append(c)
            elif args.include_positive:
                candidates.append(label_candidate_id)  # positive candidate
                for c in mentions[mention_id]["tfidf_candidates"]:
                    if c != label_candidate_id and len(candidates) < 10:
                        candidates.append(c)
            elif args.use_dense_candidates:
                if retrieval_model is None:
                    raise ValueError("`retrieval_model` parameter cannot be None")
                mention_window_for_retrieval, _, _ = get_mention_window_for_retrieval(mention_id,
                                                                                      mentions,
                                                                                      docs,
                                                                                      max_seq_length,
                                                                                      tokenizer)

                candidate_indices = get_dense_retrieval_candidates(mention_window_for_retrieval)
                # Append the retrieved candidates to the list candidates
                for i, c_idx in enumerate(candidate_indices):
                    c = all_entities[c_idx]
                    if c == label_candidate_id:
                        if i not in position_of_positive:
                            position_of_positive[i] = 1
                        else:
                            position_of_positive[i] += 1
                    if len(candidates) < args.num_candidates:
                        candidates.append(c)

        random.shuffle(candidates)
        # Target candidate
        if label_candidate_id in candidates:
            label_id = [candidates.index(label_candidate_id)]
        else:
            label_id = [-100]  # when target candidate not in candidate set

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

        if ex_index < 1:
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

    logger.info("***Postition of the positive candidate in retrieval***")
    print(position_of_positive)

    return features
