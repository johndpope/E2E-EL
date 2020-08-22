import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy

from modeling_bert import BertPreTrainedModel
from modeling_bert import BertModel

import pdb

class PreDualEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)


class DualEncoderBert(BertPreTrainedModel):
    def __init__(self, config, pretrained_bert):
        super().__init__(config)
        self.bert_mention = pretrained_bert
        self.bert_candidate = copy.deepcopy(pretrained_bert)

    def forward(self,
                args,
                mention_token_ids=None,
                mention_token_masks=None,
                candidate_token_ids_1=None,
                candidate_token_masks_1=None,
                candidate_token_ids_2=None,
                candidate_token_masks_2=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                all_candidate_embeddings=None,):

        mention_outputs = self.bert_mention.bert(
            input_ids=mention_token_ids,
            attention_mask=mention_token_masks,
        )
        pooled_mention_outputs = mention_outputs[1]

        ''' For random negative training and  For tf-idf candidates based training and evaluation'''
        if all_candidate_embeddings is None and candidate_token_ids_2 is None:
            b_size, n_c, seq_len = candidate_token_ids_1.size()
            candidate_token_ids_1 = candidate_token_ids_1.reshape(-1, seq_len)  # BC X L
            candidate_token_masks_1 = candidate_token_masks_1.reshape(-1, seq_len)  # BC X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids_1,
                attention_mask=candidate_token_masks_1,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            pooled_mention_outputs = pooled_mention_outputs.unsqueeze(1)  # B X 1 X d
            pooled_candidate_outputs = pooled_candidate_outputs.reshape(b_size, n_c, -1)  # B X C X d

            logits = torch.bmm(pooled_mention_outputs, pooled_candidate_outputs.transpose(1, 2))  # B X 1 X C
            logits = logits.reshape(b_size, n_c)  # B X C

            labels = labels.reshape(-1)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # For hard and random negative training'''
        elif candidate_token_ids_1 is not None and candidate_token_ids_2 is not None:
            # Concatenate hard negative candidates with random negatives
            candidate_token_ids = torch.cat([candidate_token_ids_1, candidate_token_ids_2], dim=1)
            candidate_token_masks = torch.cat([candidate_token_masks_1, candidate_token_masks_2], dim=1)

            b_size, n_c, seq_len = candidate_token_ids.size()

            # Mask off the padding candidates (because there maybe less than 'num_candiadtes' hard negatives)
            candidate_mask = torch.sum(candidate_token_ids, dim=2)  # B X C
            non_zeros = torch.where(candidate_mask > 0)
            candidate_mask[non_zeros] = 1  # B X C
            candidate_mask = candidate_mask.float()

            candidate_token_ids = candidate_token_ids.reshape(-1, seq_len)  # BC X L
            candidate_token_masks = candidate_token_masks.reshape(-1, seq_len)  # BC X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids,
                attention_mask=candidate_token_masks,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            pooled_mention_outputs = pooled_mention_outputs.unsqueeze(1)  # B X 1 X d
            pooled_candidate_outputs = pooled_candidate_outputs.reshape(b_size, n_c, -1)  # B X C X d

            logits = torch.bmm(pooled_mention_outputs, pooled_candidate_outputs.transpose(1, 2))  # B X 1 X C
            logits = logits.reshape(b_size, n_c)  # B X C

            # Mask off the padding candidates
            logits = logits - (1.0 - candidate_mask) * 1e31

            labels = labels.reshape(-1)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)


        else: # Evaluation with all candidates
            b_size = mention_token_ids.size(0)
            n_c, _ = all_candidate_embeddings.size()

            pooled_mention_outputs = pooled_mention_outputs.unsqueeze(1)  # B X 1 X d
            all_candidate_embeddings = all_candidate_embeddings.unsqueeze(0).expand(b_size, -1, -1) # B X C_all X d

            logits = torch.bmm(pooled_mention_outputs, all_candidate_embeddings.transpose(1, 2))  # B X 1 X C_all
            logits = logits.reshape(b_size, n_c)  # B X C_all

            labels = labels.reshape(-1)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)


        outputs = (loss, ) + (logits, )

        return outputs  # (loss), logits,