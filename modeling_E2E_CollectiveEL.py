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
        self.num_tags = 3
        self.bert_mention = pretrained_bert
        self.bert_candidate = copy.deepcopy(pretrained_bert)
        self.hidden_size = config.hidden_size
        self.mlp = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.tagger_layer = nn.Linear(config.hidden_size, self.num_tags) # 'BIO'
        self.init_modules()
        self.loss_fn_linker = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fn_tagger = nn.CrossEntropyLoss(ignore_index=-100)

    def init_modules(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        for module in self.tagger_layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

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
                seq_tags=None,
                all_candidate_embeddings=None,):

        if mention_token_ids is not None:
            mention_outputs = self.bert_mention.bert(
                input_ids=mention_token_ids,
                attention_mask=mention_token_masks,
            )
            last_hidden_states = mention_outputs[0]

        if mention_start_indices is not None and mention_end_indices is not None:
            # Pool the mention representations
            mention_start_indices = mention_start_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            mention_end_indices = mention_end_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            mention_start_embd = last_hidden_states.gather(1, mention_start_indices)
            mention_end_embd = last_hidden_states.gather(1, mention_end_indices)

            mention_embeddings = self.mlp(torch.cat([mention_start_embd, mention_end_embd], dim=2))
            mention_embeddings = mention_embeddings.reshape(-1, 1, self.hidden_size)

            # mention_embeddings = mention_start_embd.reshape(-1, 1, self.hidden_size)

        tag_logits = self.tagger_layer(last_hidden_states)  # B X L X H --> B X L X num_tags
        if seq_tags is not None:
            tag_logits = tag_logits.reshape(-1, self.num_tags) # BL X num_tags
            seq_tags = seq_tags.reshape(-1)
            tagger_loss = self.loss_fn_tagger(tag_logits, seq_tags) # BL
            # Normalize the loss
            tagger_loss = tagger_loss / seq_tags.size(0)
        else:
            return tag_logits

        if candidate_token_ids_1 is not None:
            b_size, n_c, seq_len = candidate_token_ids_1.size()
            candidate_token_ids_1 = candidate_token_ids_1.reshape(-1, seq_len)  # B(N*C) X L
            candidate_token_masks_1 = candidate_token_masks_1.reshape(-1, seq_len)  # B(N*C) X L

            candidate_outputs = self.bert_candidate.bert(
                input_ids=candidate_token_ids_1,
                attention_mask=candidate_token_masks_1,
            )
            pooled_candidate_outputs = candidate_outputs[1]

            candidate_embeddings = pooled_candidate_outputs.reshape(-1, args.num_candidates, self.hidden_size) #BN X C X H

            linker_logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
            linker_logits = linker_logits.squeeze(1)  # BN X C

            if labels is not None:
                labels = labels.reshape(-1)  # BN
                linking_loss = self.loss_fn_linker(linker_logits, labels)
                # Normalize the loss
                linking_loss = linking_loss / args.num_max_mentions
            else:
                linking_loss = None

            if self.training:
                loss = tagger_loss + linking_loss
            else:
                loss = linking_loss

            return loss, linker_logits

        # When a second set of candidates is present
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

        if all_candidate_embeddings is not None:
            b_size = mention_embeddings.size(0)
            all_candidate_embeddings = all_candidate_embeddings[0].unsqueeze(0).expand(b_size, -1, -1)  # B X C_all X H
            linker_logits = torch.bmm(mention_embeddings, all_candidate_embeddings.transpose(1, 2))
            linker_logits = linker_logits.squeeze(1)  # BN X C
            return None, linker_logits
