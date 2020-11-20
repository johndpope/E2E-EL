import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy
from modeling_bert import BertPreTrainedModel
from modeling_bert import BertModel, BertForPreTraining
import pdb


class PreDualEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # self.bert = BertForPreTraining(config)


class DualEncoderBert(BertPreTrainedModel):
    def __init__(self, config, pretrained_bert):
        super().__init__(config)
        self.config = config
        self.num_tags = 3
        self.bert_mention = pretrained_bert
        self.bert_candidate = copy.deepcopy(pretrained_bert)
        self.hidden_size = config.hidden_size
        self.bound_classifier = nn.Linear(config.hidden_size, 3)
        self.init_modules()
        self.loss_fn_linker = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fn_tagger = nn.CrossEntropyLoss(ignore_index=-100)

    def init_modules(self):
        for module in self.bound_classifier.modules():
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
                labels=None,
                seq_tags=None,
                all_candidate_embeddings=None,
                mode=None,
                ):
        if mention_token_ids is not None:
            mention_outputs = self.bert_mention.bert(
                input_ids=mention_token_ids,
                attention_mask=mention_token_masks,
            )
            last_hidden_states = mention_outputs[0]

        if mode == 'ner':
            logits = self.bound_classifier(last_hidden_states)  # B X L X H --> B X L X 3
            start_scores, end_scores, mention_scores = logits.split(1, dim=-1)
            start_scores = start_scores.squeeze(-1)
            end_scores = end_scores.squeeze(-1)
            mention_scores = mention_scores.squeeze(-1)
            # Exclude The masked tokens from start or end mentions
            start_scores[torch.where(mention_token_masks == 0)] = -float("Inf")
            end_scores[torch.where(mention_token_masks == 0)] = -float("Inf")
            mention_scores[torch.where(mention_token_masks == 0)] = -float("Inf")

            cumulative_mention_scores = torch.zeros(mention_token_masks.size()).to(mention_token_masks.device)
            cumulative_mention_scores[torch.where(mention_token_masks == 0)] = -float("Inf")
            cumulative_mention_scores[:, 0] = mention_scores[:, 0]
            for i in range(1, mention_token_masks.size(1)):
                cumulative_mention_scores[:, i] = cumulative_mention_scores[:, i-1] + mention_scores[:, i]

            all_span_mention_scores = cumulative_mention_scores.unsqueeze(1) - cumulative_mention_scores.unsqueeze(2)
            all_span_mention_scores += mention_scores.unsqueeze(2).expand_as(all_span_mention_scores)

            # Add the start scores and end scores
            all_span_start_end_scores = start_scores.unsqueeze(2) + end_scores.unsqueeze(1)

            # Add the mention span scores with the sum of start scores and end scores
            all_span_scores = all_span_start_end_scores + all_span_mention_scores

            # Mention end cannot have a higher index than mention start
            impossible_mask = torch.zeros((all_span_scores.size(1), all_span_scores.size(2)), dtype=torch.int32).to(mention_token_masks.device)
            possible_rows, possible_cols = torch.triu_indices(all_span_scores.size(1), all_span_scores.size(2))
            impossible_mask[(possible_rows, possible_cols)] = 1
            impossible_mask = impossible_mask.unsqueeze(0).expand_as(all_span_scores)
            all_span_scores[torch.where(impossible_mask == 0)] = -float("Inf")
            # Convert the scores the log-likelihoods
            all_span_logprobs = torch.sigmoid(all_span_scores).log()  # Dim: B X L X L

            # Spans with logprobs  minus "Inf" are invalid
            all_spans = torch.where(all_span_logprobs > -float("Inf"))
            all_doc_indices = all_spans[0]
            all_start_indices = all_spans[1]
            all_end_indices = all_spans[2]

            span_lengths = all_end_indices - all_start_indices + 1
            valid_span_indices = torch.where(span_lengths <= args.max_mention_length)
            valid_doc_indices = all_doc_indices[valid_span_indices]
            valid_start_indices = all_start_indices[valid_span_indices]
            valid_end_indices = all_end_indices[valid_span_indices]

            valid_spans = torch.stack((valid_doc_indices, valid_start_indices, valid_end_indices), dim=0)

            pred_spans = set()
            for i in range(valid_spans.size(1)):
                pred_spans.add((valid_spans[0][i].item(), valid_spans[1][i].item(), valid_spans[2][i].item()))

            gt_spans = set()
            gt_start_end_indices = torch.stack([mention_start_indices, mention_end_indices], dim=1)
            for i in range(gt_start_end_indices.size(0)):
                for j in range(gt_start_end_indices.size(-1)):
                    gt_spans.add((i, gt_start_end_indices[i][0][j].item(), gt_start_end_indices[i][1][j].item()))

            if self.training:
                ner_loss = 0
                # ylog(p) + (1-y)log(1-p)
                for (i, s_idx, e_idx) in pred_spans:
                    if (i, s_idx, e_idx) in gt_spans:
                        ner_loss = ner_loss - all_span_logprobs[i, s_idx, e_idx]
                    else:
                        ner_loss = ner_loss - torch.log(1 - torch.exp(all_span_logprobs[i, s_idx, e_idx]))
                ner_loss = ner_loss / len(pred_spans)
                return ner_loss
            else:
                valid_span_logprobs = all_span_logprobs[(valid_doc_indices, valid_span_indices, valid_end_indices)]
                return valid_start_indices, valid_end_indices, valid_span_logprobs

        if mode == 'ned':
            if mention_start_indices is not None and mention_end_indices is not None:

                # Pool the mention representations
                mention_embeddings = []
                for i in range(mention_start_indices.size(0)):
                    for j in range(mention_start_indices.size(1)):
                        s_idx = mention_start_indices[i][j].item()
                        e_idx = mention_end_indices[i][j].item()
                        m_embd = torch.mean(last_hidden_states[:, s_idx:e_idx + 1, :], dim=1)
                        mention_embeddings.append(m_embd)
                mention_embeddings = torch.cat(mention_embeddings, dim=0).unsqueeze(1)

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

                linker_logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
                linker_logits = linker_logits.squeeze(1)  # BN X C

                if labels is not None:
                    labels = labels.reshape(-1)  # BN
                    linking_loss = self.loss_fn_linker(linker_logits, labels)
                    # Normalize the loss
                    num_mentions = torch.where(labels >= 0)[0].size(0)
                    linking_loss = linking_loss / num_mentions
                else:
                    linking_loss = None

                return linking_loss, linker_logits

            # For hard and random negative training
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

                # Mask off the padding candidates (because there maybe less than 'num_candidates' hard negatives)
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

                candidate_embeddings = pooled_candidate_outputs.reshape(-1, 2 * args.num_candidates,
                                                                        self.hidden_size)  # BN X 2*C X H

                linker_logits = torch.bmm(mention_embeddings, candidate_embeddings.transpose(1, 2))
                linker_logits = linker_logits.squeeze(1)  # BN X C

                # logits = logits.reshape(b_size, n_c)  # B X C

                # Mask off the padding candidates
                candidate_mask = candidate_mask.reshape(-1, 2 * args.num_candidates)
                linker_logits = linker_logits - (1.0 - candidate_mask) * 1e31

                labels = labels.reshape(-1)

                linker_loss = self.loss_fn_linker(linker_logits, labels)
                # Normalize the loss
                num_mentions = torch.where(labels >= 0)[0].size(0)
                linker_loss = linker_loss / num_mentions
                return linker_loss, linker_logits

            if all_candidate_embeddings is not None:
                b_size = mention_embeddings.size(0)
                all_candidate_embeddings = all_candidate_embeddings[0].unsqueeze(0).expand(b_size, -1, -1)  # B X C_all X H
                linker_logits = torch.bmm(mention_embeddings, all_candidate_embeddings.transpose(1, 2))
                linker_logits = linker_logits.squeeze(1)  # BN X C
                return None, linker_logits
