import os
import json
import random
import math
import pdb
import logging
logger = logging.getLogger(__name__)
import faiss
import torch

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
    window = prefix + ['[Ms]'] + mention + ['[Me]'] +suffix
    window = window[:max_size]  # Truncate tail of suffix

    mention_start_index = len(prefix)
    mention_end_index = len(prefix) + len(mention) - 1

    return window, mention_start_index, mention_end_index


def get_mention_window(mention_id, mentions, docs,  max_seq_length, tokenizer):
    max_len_context = max_seq_length - 2 # number of characters
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


def get_entity_window(entity_text, max_entity_len, tokenizer):
    entity_tokens = tokenizer.tokenize(entity_text)
    if len(entity_tokens) > max_entity_len:
        entity_tokens = entity_tokens[:max_entity_len]
    return entity_tokens


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, mention_token_ids, mention_token_masks, candidate_token_ids, candidate_token_masks, label_ids):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        self.candidate_token_ids = candidate_token_ids
        self.candidate_token_masks = candidate_token_masks
        self.label_ids = label_ids

def convert_examples_to_features(
    mentions,
    docs,
    entities,
    max_seq_length,
    tokenizer,
    args,
    model=None,
):

    # All entities
    all_entities = list(entities.keys())
    all_entity_token_ids = []
    all_entity_token_masks = []

    if args.use_all_candidates:
        for c_idx, c in enumerate(all_entities):
            entity_text = entities[c]
            max_entity_len = max_seq_length  # Number of tokens
            entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
            # [CLS] candidate text [SEP]
            candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
            candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
            if len(candidate_tokens) > max_seq_length:
                candidate_tokens = candidate_tokens[:max_seq_length]
                candidate_masks = [1] * max_seq_length
            else:
                candidate_len = len(candidate_tokens)
                pad_len = max_seq_length - candidate_len
                candidate_tokens += [tokenizer.pad_token_id] * pad_len
                candidate_masks = [1] * candidate_len + [0] * pad_len

            assert len(candidate_tokens) == max_seq_length
            assert len(candidate_masks) == max_seq_length

            all_entity_token_ids.append(candidate_tokens)
            all_entity_token_masks.append(candidate_masks)

    if model is not None:
        # Gather all candidate embeddings for hard negative mining
        all_candidate_embeddings = []
        with torch.no_grad():
            # Forward pass through the candidate encoder of the dual encoder
            for i, (entity_tokens, entity_tokens_masks) in enumerate(
                    zip(all_entity_token_ids, all_entity_token_masks)):
                candidate_token_ids = torch.LongTensor([entity_tokens]).to(args.device)
                candidate_token_masks = torch.LongTensor([entity_tokens_masks]).to(args.device)
                candidate_outputs = model.bert_candidate.bert(
                    input_ids=candidate_token_ids,
                    attention_mask=candidate_token_masks,
                )
                candidate_embedding = candidate_outputs[1]
                all_candidate_embeddings.append(candidate_embedding)

        all_candidate_embeddings = torch.cat(all_candidate_embeddings, dim=0)

        # Indexing for faster search (using FAISS)
        d = all_candidate_embeddings.size(1)
        all_candidate_index = faiss.IndexFlatL2(d)  # build the index, d=size of vectors
        # here we assume xb contains a n-by-d numpy matrix of type float32
        all_candidate_embeddings = all_candidate_embeddings.cpu().numpy()
        all_candidate_index.add(all_candidate_embeddings)

    # Process the mentions
    features = []
    for (ex_index, mention_id) in enumerate(mentions.keys()):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(mentions))

        mention_window, mention_start_index, mention_end_index = get_mention_window(mention_id,
                                                                            mentions,
                                                                            docs,
                                                                            max_seq_length,
                                                                            tokenizer)

        # [CLS] mention with context [SEP]
        mention_tokens = [tokenizer.cls_token] + mention_window + [tokenizer.sep_token]
        mention_tokens = tokenizer.convert_tokens_to_ids(mention_tokens)
        if len(mention_tokens) > max_seq_length:
            mention_tokens = mention_tokens[:max_seq_length]
            mention_tokens_mask = [1] * max_seq_length
        else:
            mention_len = len(mention_tokens)
            pad_len = max_seq_length - mention_len
            mention_tokens += [tokenizer.pad_token_id] * pad_len
            mention_tokens_mask = [1] * mention_len + [0] * pad_len

        assert len(mention_tokens) == max_seq_length
        assert len(mention_tokens_mask) == max_seq_length

        # Build list of candidates
        label_candidate_id = mentions[mention_id]['label_candidate_id']
        candidates = []
        if args.do_train:
            candidates.append(label_candidate_id)  # positive candidate
            if model is None:
                # If model is not given, then use either random negative candidates
                # or tfidf candidates based on the choice
                if args.random_candidates:  # Random negatives
                    candidate_pool = set(entities.keys()) - set([label_candidate_id])
                    negative_candidates = random.sample(candidate_pool, 9)
                    candidates += negative_candidates
                elif args.tfidf_candidates: # TF-IDF negatives
                    for c in mentions[mention_id]["tfidf_candidates"]:
                        if c != label_candidate_id and len(candidates) < 10:
                            candidates.append(c)
            else:
                # TODO:Implement hard negative candidate mining
                print("Performing hard negative candidate mining ...")
                mention_tokens = torch.LongTensor([mention_tokens]).to(args.devide)
                mention_tokens_mask = torch.LongTensor([mention_tokens_mask]).to(args.devide)
                # Forward pass through the mention encoder of the dual encoder
                mention_outputs = model.bert_mention.bert(
                    input_ids=mention_tokens,
                    attention_mask=mention_tokens_mask,
                )
                mention_embedding = mention_outputs[1]  # 1 X d
                mention_embedding = mention_embedding.cpu().numpy()

                # Perform similarity search
                distance, candidate_indices = all_candidate_index.search(mention_embedding, 10)
                candidate_indices = candidate_indices[0]  # original size 1 X 10 -> 10

                # Append the hard negative candidates to the list of all candidates
                for c_idx in  candidate_indices:
                    c = all_entities[c_idx]
                    if c != label_candidate_id and len(candidates) < 10:
                        candidates.append(c)

        elif args.do_eval:
            if args.include_positive:
                candidates.append(label_candidate_id)  # positive candidate
                for c in mentions[mention_id]["tfidf_candidates"]:
                    if c != label_candidate_id and len(candidates) < 10:
                        candidates.append(c)
            elif args.tfidf_candidates:
                for c in mentions[mention_id]["tfidf_candidates"]:
                    candidates.append(c)
            elif args.use_all_candidates:
                candidates = all_entities

        if not args.use_all_candidates:
            random.shuffle(candidates)

        if args.use_all_candidates:
            # If all candidates are considered during inference,
            # then place dummy candidate tokens and candidate masks
            candidate_token_ids = None
            candidate_token_masks = None
        else:
            candidate_token_ids = []
            candidate_token_masks = []

            for c_idx, c in enumerate(candidates):
                entity_text = entities[c]
                max_entity_len = max_seq_length  # Number of tokens
                entity_window = get_entity_window(entity_text, max_entity_len, tokenizer)
                # [CLS] candidate text [SEP]
                candidate_tokens = [tokenizer.cls_token] + entity_window + [tokenizer.sep_token]
                candidate_tokens = tokenizer.convert_tokens_to_ids(candidate_tokens)
                if len(candidate_tokens) > max_seq_length:
                    candidate_tokens = candidate_tokens[:max_seq_length]
                    candidate_masks = [1] * max_seq_length
                else:
                    candidate_len = len(candidate_tokens)
                    pad_len = max_seq_length - candidate_len
                    candidate_tokens += [tokenizer.pad_token_id] * pad_len
                    candidate_masks = [1] * candidate_len + [0] * pad_len

                assert len(candidate_tokens) == max_seq_length
                assert len(candidate_masks) == max_seq_length

                candidate_token_ids.append(candidate_tokens)
                candidate_token_masks.append(candidate_masks)

        # Target candidate
        if label_candidate_id in candidates:
            label_id = [candidates.index(label_candidate_id)]
        else:
            label_id = [-100]  # when target candidate not in candidate set

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("mention_token_ids: %s", " ".join([str(x) for x in mention_tokens]))
            logger.info("mention_token_masks: %s", " ".join([str(x) for x in mention_tokens_mask]))
            if candidate_token_ids is not None:
                logger.info("candidate_token_ids: %s", " ".join([str(x) for x in candidate_token_ids]))
                logger.info("candidate_token_masks: %s", " ".join([str(x) for x in candidate_token_masks]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_id]))

        features.append(
            InputFeatures(mention_token_ids=mention_tokens,
                          mention_token_masks=mention_tokens_mask,
                          candidate_token_ids=candidate_token_ids,
                          candidate_token_masks=candidate_token_masks,
                          label_ids=label_id,
                          )
        )
    return features, (all_entities, all_entity_token_ids, all_entity_token_masks)
