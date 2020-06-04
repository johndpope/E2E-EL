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
        self.loss_fn = nn.CrossEntropyLoss()

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

        if mention_token_ids is not None:
            mention_outputs = self.bert_mention.bert(
                input_ids=mention_token_ids,
                attention_mask=mention_token_masks,
            )
            last_hidden_states = mention_outputs[0]

            return last_hidden_states

        if candidate_token_ids_1 is not None:
            b_size, n_c, seq_len = candidate_token_ids_1.size()
            candidate_token_ids_1 = candidate_token_ids_1.reshape(-1, seq_len)  # BC X L
            candidate_token_masks_1 = candidate_token_masks_1.reshape(-1, seq_len)  # BC X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids_1,
                attention_mask=candidate_token_masks_1,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            return pooled_candidate_outputs

        # When a seocnd set of candidates is present
        if candidate_token_ids_2 is not None:
            b_size, n_c, seq_len = candidate_token_ids_2.size()

            # Mask off the padding candidates
            candidate_mask = torch.sum(candidate_token_ids_2, dim=2)  # B X C
            non_zeros = torch.where(candidate_mask > 0)
            candidate_mask[non_zeros] = 1  # B X C

            candidate_token_ids_2 = candidate_token_ids_2.reshape(-1, seq_len)  # BC X L
            candidate_token_masks_2 = candidate_token_masks_2.reshape(-1, seq_len)  # BC X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids_2,
                attention_mask=candidate_token_masks_2,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            return pooled_candidate_outputs
