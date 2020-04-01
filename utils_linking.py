import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
logger = logging.getLogger(__name__)

word_piece_ment = 128
word_piece_ent = 128

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


        # if self.phase_test:
        #     self.ments_test = {}
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("mentions test dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.ments_test[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #         print("mentions test dataset is done :)")
        # elif self.phase_val:
        #     self.ments_val = {}
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("mentions val dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.ments_val[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #         print("mentions val dataset is done :)")
        # else:
        #     self.ments_train = {}
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("mentions train dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.ments_train[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #         print("mentions train datast is done :)")

        # if self.phase_test:
        #     self.cand_test = {}
        #     self.cand_test_entity = []
        #     self.cand_test_ments_id = []
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("candidates test dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.cand_test[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #             self.cand_test_ments_id.append([fields["mention_id"], len(fields["tfidf_candidates"])])
        #             for ent in fields["tfidf_candidates"]:
        #                 self.cand_test_entity.append([fields["mention_id"], ent])
        #         print("candidates test dataset is done :)")
        # elif self.phase_val:
        #     self.cand_val = {}
        #     self.cand_val_ments_id = []
        #     self.cand_val_entity = []
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("candidates val dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.cand_val[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #             self.cand_val_ments_id.append([fields["mention_id"],len(fields["tfidf_candidates"])])
        #             for ent in fields["tfidf_candidates"]:
        #                 self.cand_val_entity.append([fields["mention_id"], ent])
        #         print("candidates val dataset is done :)")
        # else:
        #     self.cand_train = {}
        #     self.cand_train_ments_id = []
        #     self.cand_train_entity = []
        #     with open(os.path.join(data_dir, mode, 'mentions/mentions.json')) as f:
        #         print("candidates train dataset is loading......")
        #         for line in f:
        #             fields = json.loads(line)
        #             self.cand_train[fields["mention_id"]] = {k: v for k, v in fields.items() if k != "mention_id"}
        #             self.cand_train_ments_id.append([fields["mention_id"], len(fields["tfidf_candidates"])])
        #             for ent in fields["tfidf_candidates"]:
        #                 self.cand_train_entity.append([fields["mention_id"], ent])
        #         print("candidates train dataset is done :)")

        # self.all_mentions = []
        # for mention_id in self.ments:
        #     self.all_mentions.append([mention_id, self.ments[mention_id]['tfidf_candidates']])

    # def __len__(self):
    #     """get the size of dataset"""
    #     return len(self.all_mentions) # number_of_mentions
    #
    #     # if self.phase_test:
    #     #     return len(self.cand_test_ments_id)
    #     # elif self.phase_val:
    #     #     return len(self.cand_val_ments_id)
    #     # else:
    #     #     return len(self.cand_train_ments_id)

    # def __getitem__(self, item):
    #     """
    #     :return batch_tokens_pair_pad: (max_seq_len * input_size_k), k means the number of candidates
    #             label_candidate_id_k: one-hot
    #             sentence_len_k: original length of sentence before padding
    #             idx: index of mention boundary
    #             batch_type_token_ids_pad: token_type_id mask of sentence (0, 0, ..., 0, 1, 1, ..., 1)
    #     """
    #
    #     # if self.phase_test:
    #     #     batch_tokens_pair = []
    #     #     batch_tokens_pair_pad = []
    #     #     label_candidate_id_k = []
    #     #     idx = []
    #     #     mention_id, mention_k = self.cand_test_ments_id[item]
    #     #     cand_test_ent_k = [x for x in self.cand_test_entity if x[0] == mention_id]
    #     #
    #     #     for ent_id in range(len(cand_test_ent_k)):
    #     #         candidate_id = cand_test_ent_k[ent_id][-1]
    #     #         start_index = self.ments_test[mention_id]["start_index"]
    #     #         end_index = self.ments_test[mention_id]["end_index"]
    #     #         context_document_id = self.ments_test[mention_id]["context_document_id"]
    #     #         label_candidate_id = self.ments_test[mention_id]["label_candidate_id"]
    #     #
    #     #         context_text = self.docs[context_document_id]["text"]
    #     #         entity_text = self.entities[candidate_id]
    #     #         mention = context_text[start_index:end_index + 1]
    #     #         # label_title = self.docs[label_document_id]["title"]
    #     #         label_text = self.docs[label_document_id]["text"]
    #     #
    #     #         ment_text = truncate_ment(context_text, word_piece_ment, start_index, end_index)
    #     #         desc_text = truncate_ent(entity_text, word_piece_ent)
    #     #
    #     #         tokens_pair = torch.tensor(self.tokenizer.encode(ment_text, desc_text, add_special_tokens=True))
    #     #         batch_tokens_pair.append(tokens_pair)
    #     #         if candidate_id == label_candidate_id:
    #     #             label_candidate_id_k.append(1)
    #     #         else:
    #     #             label_candidate_id_k.append(0)
    #     #
    #     # elif self.phase_val:
    #     #     batch_tokens_pair = []
    #     #     batch_tokens_pair_pad = []
    #     #     batch_type_token_ids = []
    #     #     batch_type_token_ids_pad = []
    #     #     label_document_id_k = []
    #     #     idx = []
    #     #     mention_id, mention_k = self.cand_val_ments_id[item]
    #     #     cand_val_ent_k = [x for x in self.cand_val_entity if x[0] == mention_id]
    #     #
    #     #     for ent_id in range(len(cand_val_ent_k)):
    #     #         candidate_id = cand_val_ent_k[ent_id][-1]
    #     #         start_index = self.ments_val[mention_id]["start_index"]
    #     #         end_index = self.ments_val[mention_id]["end_index"]
    #     #         context_document_id = self.ments_val[mention_id]["context_document_id"]
    #     #         label_document_id = self.ments_val[mention_id]["label_document_id"]
    #     #
    #     #         context_text = self.docs[context_document_id]["text"]
    #     #         entity_text = self.docs[candidate_id]["text"]
    #     #         mention = context_text[start_index:end_index + 1]
    #     #         label_title = self.docs[label_document_id]["title"]
    #     #         label_text = self.docs[label_document_id]["text"]
    #     #
    #     #         ment_text = truncate_ment(context_text, word_piece_ment, start_index, end_index)
    #     #         desc_text = truncate_ent(entity_text, word_piece_ent)
    #     #
    #     #         mention_text = ' '.join(mention)
    #     #         token_mention = self.tokenizer.encode(mention_text)
    #     #
    #     #         ment_text_str = ' '.join(ment_text)
    #     #         desc_text_str = ' '.join(desc_text)
    #     #         ment_text_id = self.tokenizer.encode(ment_text_str)
    #     #         desc_text_id = self.tokenizer.encode(desc_text_str)
    #     #         tokens_pair = self.tokenizer.build_inputs_with_special_tokens(ment_text_id, token_ids_1=desc_text_id)
    #     #         token_type_id = self.tokenizer.create_token_type_ids_from_sequences(ment_text_id, token_ids_1=desc_text_id)
    #     #         idx = []
    #     #         for i in token_mention:
    #     #             idx.append(tokens_pair.index(i))
    #     #
    #     #         batch_tokens_pair.append(torch.tensor(tokens_pair))
    #     #         batch_type_token_ids.append(torch.tensor(token_type_id))
    #     #         if candidate_id == label_document_id:
    #     #             label_document_id_k.append(1)
    #     #         else:
    #     #             label_document_id_k.append(0)
    #     #
    #     # else:
    #     #     batch_tokens_pair = []
    #     #     batch_tokens_pair_pad = []
    #     #     batch_type_token_ids = []
    #     #     batch_type_token_ids_pad = []
    #     #     label_document_id_k = []
    #     #     idx = []
    #     #     mention_id, mention_k = self.cand_train_ments_id[item]
    #     #     cand_train_ent_k = [x for x in self.cand_train_entity if x[0] == mention_id]
    #     #
    #     #     for ent_id in range(len(cand_train_ent_k)):
    #     #         candidate_id = cand_train_ent_k[ent_id][-1]
    #     #         start_index = self.ments_train[mention_id]["start_index"]
    #     #         end_index = self.ments_train[mention_id]["end_index"]
    #     #         context_document_id = self.ments_train[mention_id]["context_document_id"]
    #     #         label_document_id = self.ments_train[mention_id]["label_document_id"]
    #     #
    #     #         context_text = self.docs[context_document_id]["text"]
    #     #         entity_text = self.docs[candidate_id]["text"]
    #     #         mention = context_text[start_index:end_index + 1]
    #     #         label_title = self.docs[label_document_id]["title"]
    #     #         label_text = self.docs[label_document_id]["text"]
    #     #
    #     #         ment_text = truncate_ment(context_text, word_piece_ment, start_index, end_index)
    #     #         desc_text = truncate_ent(entity_text, word_piece_ent)
    #     #
    #     #         mention_text = ' '.join(mention)
    #     #         token_mention = self.tokenizer.encode(mention_text)
    #     #
    #     #         ment_text_str = ' '.join(ment_text)
    #     #         desc_text_str = ' '.join(desc_text)
    #     #         ment_text_id = self.tokenizer.encode(ment_text_str)
    #     #         desc_text_id = self.tokenizer.encode(desc_text_str)
    #     #         tokens_pair = self.tokenizer.build_inputs_with_special_tokens(ment_text_id, token_ids_1=desc_text_id)
    #     #         token_type_id = self.tokenizer.create_token_type_ids_from_sequences(ment_text_id, token_ids_1=desc_text_id)
    #     #         idx = []
    #     #         for i in token_mention:
    #     #             idx.append(tokens_pair.index(i))
    #     #
    #     #         batch_tokens_pair.append(torch.tensor(tokens_pair))
    #     #         batch_type_token_ids.append(torch.tensor(token_type_id))
    #     #         if candidate_id == label_document_id:
    #     #             label_document_id_k.append(1)
    #     #         else:
    #     #             label_document_id_k.append(0)
    #
    #     sentence_len_k = [list(y.shape) for y in batch_tokens_pair]
    #     sentence_len_k_num = [z[0] for z in sentence_len_k]
    #     sentence_len_k_num.sort(reverse=True)
    #     max_len = sentence_len_k_num[0]
    #
    #     for i in range(len(batch_tokens_pair)):
    #         pad_len = max_len - list(batch_tokens_pair[i].shape)[0]
    #         sent_pad = F.pad(batch_tokens_pair[i], (0, pad_len), mode='constant', value=0.)
    #         type_id_pad = F.pad(batch_type_token_ids[i], (0, pad_len), mode='constant', value=0.)
    #         batch_tokens_pair_pad.append(sent_pad.long())
    #         batch_type_token_ids_pad.append(type_id_pad)
    #
    #     return batch_tokens_pair_pad, label_document_id_k, sentence_len_k, idx, batch_type_token_ids_pad


def truncate_ment(mention_text, length, start_idx, end_idx):
    """truncate the mention documents with 'word_piece_ment' word pieces (default = 128) that surround the mention,
    and make sure that each sentence has the same length."""
    mention_text = mention_text.split(' ')
    text_length = len(mention_text)
    mention_length = end_idx - start_idx + 1
    t_start_idx = int(start_idx - length / 2)
    r_half_length = int(length / 2 - mention_length)
    t_end_idx = int(end_idx + r_half_length + 1)

    ment_text = mention_text[t_start_idx: t_end_idx]

    if text_length <= length:
        start = 0
        end = text_length
        ment_text = mention_text[start: end]
    else:
        if t_start_idx < 0:
            f_half_length = abs(t_start_idx)
            start = 0
            end = t_end_idx + f_half_length
            ment_text = mention_text[start: end]
        if t_end_idx > text_length:
            s_half_length = t_end_idx - text_length
            start = t_start_idx - s_half_length
            end = text_length
            ment_text = mention_text[start: end]

    return ment_text


def truncate_ent(description_text, length):
    """truncate the entity description with the first "k" words"""
    description_text = description_text.split(' ')
    desc_text = description_text[0:length]

    return desc_text


# def get_data_loader(docs_path, ment_path_train, ment_path_test, ment_path_val,
#                     cand_path_train, cand_path_test, cand_path_val, phase_test, phase_val,
#                     batch_size, shuffle, workers, tokenizer):
#     """build data_loader
#     :return size of each batch is (number of candidates(k = 64) * max length of sentence)
#     """
#
#     dataset_toy = PreDataset(docs_path, ment_path_train, ment_path_test, ment_path_val,
#                                            cand_path_train, cand_path_test, cand_path_val, tokenizer,
#                                            phase_test=phase_test, phase_val=phase_val)
#     data_loader = DataLoader(dataset_toy, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_fn)
#
#     return data_loader


# def collate_fn(batch_data):
#     """pad the number of candidates into 64"""
#     length_ment = []
#
#     for length in batch_data:
#         length_ment.append(len(length[0][0]))
#
#     max_length = max(length_ment)
#     if max_length > 512:
#         max_length = 512
#
#     k_list = []
#     tokens_k = []
#     labels_k = []
#     lens_k = []
#     mention_token_id = []
#     type_tokens_k = []
#     for t in batch_data:
#         mention_token_id.append(t[3])
#         k_list.append(len(t[0]))
#         pad_len = 64 - len(t[0])
#
#         for i in range(len(t[0])):
#             if len(t[0][i]) < 512:
#                 sample_tokens = F.pad(t[0][i], (0, (max_length-len(t[0][i]))), mode='constant', value=0.)
#                 tokens_k.append(sample_tokens)
#                 sample_type_tokens = F.pad(t[4][i], (0, (max_length-len(t[4][i]))), mode='constant', value=0.)
#                 type_tokens_k.append(sample_type_tokens)
#             else:
#                 sample_tokens = t[0][i][0:512]
#                 tokens_k.append(sample_tokens)
#                 sample_type_tokens = t[4][i][0:512]
#                 type_tokens_k.append(sample_type_tokens)
#
#         for sample_labels in t[1]:
#             labels_k.append(sample_labels)
#         for sample_lens in t[2]:
#             lens_k.append(sample_lens)
#         if pad_len >= 0:
#             for i in range(pad_len):
#                 tokens_k.append(torch.zeros(max_length).long())
#                 type_tokens_k.append(torch.zeros(max_length).long())
#                 labels_k.append(0)
#                 lens_k.append([0])
#
#     return tokens_k, labels_k, lens_k, mention_token_id, type_tokens_k


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_examples_to_features(
    mentions,
    docs,
    entities,
    max_seq_length,
    tokenizer,
    # cls_token_at_end=False,
    # cls_token="[CLS]",
    # cls_token_segment_id=1,
    # sep_token="[SEP]",
    # sep_token_extra=False,
    # pad_on_left=False,
    # pad_token=0,
    # pad_token_segment_id=0,
    # pad_token_label_id=-100,
    # sequence_a_segment_id=0,
    # mask_padding_with_zero=True,
):

    features = []
    for (ex_index, mention_id) in enumerate(mentions.keys()):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(mentions))

        content_document_id = mentions[mention_id]['content_document_id']
        context_text = docs[content_document_id]['text']
        start_index = mentions[mention_id]['start_index']
        end_index = mentions[mention_id]['end_index']
        label_candidate_id = mentions[mention_id]["label_candidate_id"]
        ment_text = truncate_ment(context_text, word_piece_ment, start_index, end_index)
        ment_text_str = ' '.join(ment_text)
        ment_text_id = tokenizer.encode(ment_text_str)[1:-1]
        # print(ment_text_id)

        input_ids = []
        input_mask = []
        segment_ids = []
        label_ids = []
        # print("CLS", tokenizer.convert_tokens_to_ids([tokenizer.cls_token]))
        # print("SEP", tokenizer.convert_tokens_to_ids([tokenizer.sep_token]))
        # print("PAD", tokenizer.convert_tokens_to_ids([tokenizer.pad_token]))

        for c_idx, c in enumerate(mentions[mention_id]["tfidf_candidates"]):
            entity_text = entities[c]
            entity_text = truncate_ent(entity_text, word_piece_ent)
            entity_text_str = ' '.join(entity_text)
            entity_text_id = tokenizer.encode(entity_text_str)[1:-1]
            tokens_pair = tokenizer.build_inputs_with_special_tokens(ment_text_id, token_ids_1=entity_text_id)
            # print(tokens_pair)
            token_type_id = tokenizer.create_token_type_ids_from_sequences(ment_text_id, token_ids_1=entity_text_id)
            # print(token_type_id)

            if len(tokens_pair) > max_seq_length:
                tokens_pair = tokens_pair[:max_seq_length]
                token_type_id = token_type_id[:max_seq_length]
                tokens_mask = [1]*max_seq_length
            else:
                pair_len = len(tokens_pair)
                pad_len = max_seq_length - pair_len
                tokens_pair = tokens_pair + tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
                token_type_id = token_type_id + tokenizer.convert_tokens_to_ids([tokenizer.pad_token]) * pad_len
                tokens_mask = [1]*pair_len + [0] * pad_len

            if c == label_candidate_id: # Target candidate
                label_id = [c_idx]

            assert len(tokens_pair) == max_seq_length
            assert len(token_type_id) == max_seq_length
            assert len(tokens_mask) == max_seq_length

            input_ids.append(tokens_pair)
            input_mask.append(tokens_mask)
            segment_ids.append(token_type_id)
            label_ids.append(label_id)

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("guid: %s", .guid)
            # logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids[ex_index]]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask[ex_index]]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids[ex_index]]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids[ex_index]]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          )
        )
    return features
