import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import copy

from modeling_bert import BertPreTrainedModel
from modeling_bert import BertModel


class PreDualEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)


class DualEncoderBert(BertPreTrainedModel):
    def __init__(self, config, pretrained_bert):
        super().__init__(config)
        self.num_labels = 10
        self.bert_mention = copy.deepcopy(pretrained_bert)
        self.bert_candidate = copy.deepcopy(pretrained_bert)

        self.init_weights()

    def forward(
        self,
        mention_token_ids=None,
        mention_token_masks=None,
        candidate_token_ids=None,
        candidate_token_masks=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """
        batch_size = mention_token_ids.size(0)
        batch_loss = 0
        batch_logits = []

        mention_outputs = self.bert_mention(
            input_ids=mention_token_ids,
            attention_mask=mention_token_masks,
        )
        pooled_mention_outputs = mention_outputs[1]
        print(pooled_mention_outputs.size())

        for i in range(batch_size):
            candidate_outputs = self.bert_candidate(
                input_ids=candidate_token_ids[i],
                attention_mask=candidate_token_masks[i],
            )
            pooled_candidate_outputs = candidate_outputs[1]
            print(pooled_candidate_outputs.size())

            logits = torch.mm(pooled_mention_outputs[i].reshape(1, -1), pooled_candidate_outputs.t())
            logits = logits.reshape(-1, self.num_labels)

            batch_logits.append(logits)

            if labels is not None:
                if self.num_labels == 1:
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                    batch_loss += loss
                else:
                    # We are doing classification
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels[i][0].view(-1))
                    batch_loss += loss

        outputs = (batch_loss/batch_size,) + (torch.cat(batch_logits, dim=0),)

        return outputs  # (loss), logits,