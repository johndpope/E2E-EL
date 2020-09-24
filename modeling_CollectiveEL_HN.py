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
        self.config = config
        self.bert_mention = pretrained_bert
        self.bert_candidate = copy.deepcopy(pretrained_bert)
        self.hidden_size = config.hidden_size
        self.mlp = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.init_mlp()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def init_mlp(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                args,
                mention_token_ids=None,
                mention_token_masks=None,
                mention_start_indices=None,
                mention_end_indices=None,
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

            # Pool the mention representations
            mention_start_indices = mention_start_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            mention_end_indices = mention_end_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            mention_start_embd = last_hidden_states.gather(1, mention_start_indices)
            mention_end_embd = last_hidden_states.gather(1, mention_end_indices)

            mention_embeddings = self.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
            mention_embeddings = mention_embeddings.reshape(-1, 1, self.hidden_size)

            # mention_embeddings = mention_start_embd.reshape(-1, 1, self.hidden_size)

        ''' For random negative training and  For tf-idf candidates based training and evaluation'''
        if all_candidate_embeddings is None and candidate_token_ids_2 is None:
            b_size, n_c, seq_len = candidate_token_ids_1.size()
            candidate_token_ids_1 = candidate_token_ids_1.reshape(-1, seq_len)  # B(N*C) X L
            candidate_token_masks_1 = candidate_token_masks_1.reshape(-1, seq_len)  # B(N*C) X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids_1,
                attention_mask=candidate_token_masks_1,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            candidate_embeddings = pooled_candidate_outputs.reshape(-1, args.num_candidates, self.hidden_size) #BN X C X H

            logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
            logits = logits.squeeze(1)  # BN X C

            if labels is not None:
                labels = labels.reshape(-1)  # BN
                loss = self.loss_fn(logits, labels)
                # Normalize the loss
                num_mentions = torch.where(labels >= 0)[0].size(0)
                loss = loss / num_mentions
            else:
                loss = None
            return loss, logits

        # For hard and random negative training'''
        elif candidate_token_ids_1 is not None and candidate_token_ids_2 is not None:
            # Concatenate hard negative candidates with random negatives
            b_size, _, seq_len = candidate_token_ids_1.size()
            candidate_token_ids_1 = candidate_token_ids_1.reshape(b_size, -1, args.num_candidates, seq_len)
            candidate_token_masks_1 = candidate_token_masks_1.reshape(b_size, -1, args.num_candidates, seq_len)
            candidate_token_ids_2 = candidate_token_ids_2.reshape(b_size, -1, args.num_candidates, seq_len)
            candidate_token_masks_2 = candidate_token_masks_2.reshape(b_size, -1, args.num_candidates, seq_len)

            candidate_token_ids = torch.cat([candidate_token_ids_1, candidate_token_ids_2], dim=2)
            candidate_token_masks = torch.cat([candidate_token_masks_1, candidate_token_masks_2], dim=2)

            candidate_token_ids = candidate_token_ids.reshape(b_size, -1, seq_len)
            candidate_token_masks = candidate_token_masks.reshape(b_size, -1, seq_len)

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

            candidate_embeddings = pooled_candidate_outputs.reshape(-1, 2*args.num_candidates,
                                                                    self.hidden_size)  # BN X 2*C X H

            logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
            logits = logits.squeeze(1)  # BN X C

            # logits = logits.reshape(b_size, n_c)  # B X C

            # Mask off the padding candidates
            candidate_mask = candidate_mask.reshape(-1, 2*args.num_candidates)
            logits = logits - (1.0 - candidate_mask) * 1e31

            labels = labels.reshape(-1)

            loss = self.loss_fn(logits, labels)
            # Normalize the loss
            num_mentions = torch.where(labels >= 0)[0].size(0)
            loss = loss / num_mentions
            return loss, logits

        if all_candidate_embeddings is not None:
            b_size = mention_embeddings.size(0)
            all_candidate_embeddings = all_candidate_embeddings[0].unsqueeze(0).expand(b_size, -1, -1)  # B X C_all X H
            logits = torch.bmm(mention_embeddings, all_candidate_embeddings.transpose(1, 2))
            logits = logits.squeeze(1)  # BN X C
            return None, logits
