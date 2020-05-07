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
                mention_token_ids=None,
                mention_token_masks=None,
                candidate_token_ids=None,
                candidate_token_masks=None,
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

        if all_candidate_embeddings is None:
            b_size, n_c, seq_len = candidate_token_ids.size()
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

        else:
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