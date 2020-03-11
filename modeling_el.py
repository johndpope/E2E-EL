import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from modeling_bert import BertPreTrainedModel
from modeling_bert import BertModel

class BertForEntityLinking(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.scorer = nn.CosineSimilarity()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        target_entity_ids=None,
        candidates_info=None,
        tokenizer=None,
        mode=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        target_entity_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`)
            IDs of the target candidates corresponding to each mention span. Outside the mention span, it is -100

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            NER Classification loss + EL ranking loss
        predicted_ranks (:obj:`torch.LongTensor`) of shape :obj:`(1,)`, returned when ``mode == 'eval'``
            Ranks of the target candidate entities
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
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

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # ************** <<<<<< Entity Linking >>>>>> ********************
        entities, entities_to_idx, idx_to_entities = candidates_info
        def get_candidate_embedding(candidate_name):
            tokens = tokenizer.tokenize(candidate_name)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_ids = torch.LongTensor(token_ids).cuda()
            with torch.no_grad():
                # Candidate embeddings are obtained from the input word embedding layer of BERT
                token_embeddings = self.bert.embeddings.word_embeddings(token_ids)
                candidate_embedding = token_embeddings.mean(dim=0).unsqueeze(0)
            return candidate_embedding


        # Get the mention spans --> mentions --> mention embeddings
        mention_spans = target_entity_ids != -100
        mentions = target_entity_ids[mention_spans]
        all_mention_token_embeddings = sequence_output[mention_spans]  # last hidden state of BERT

        mention_span_token_embeddings = {}
        prev_mention_id = -100
        for i in range(mentions.size(0)):
            current_mention_id = mentions[i].item()
            if current_mention_id != prev_mention_id:
                mention_span_token_embeddings[current_mention_id] = []
            mention_span_token_embeddings[current_mention_id].append(all_mention_token_embeddings[i].unsqueeze(0))
            prev_mention_id = current_mention_id

        if mode == 'train':
            el_loss = []
            for k in mention_span_token_embeddings:
                # Mention Embedding by mean pooling the embeddings of the mention span tokens
                mention_embedding = torch.cat(mention_span_token_embeddings[k], dim=0).mean(dim=0).unsqueeze(0)

                # Get the embedding of the positive candidate
                target_entity = idx_to_entities[k]
                positive_candidate = entities[target_entity]['label']
                p_c_embedding = get_candidate_embedding(positive_candidate)

                # Get the embeddings of the negative candidates
                negative_candidate_embeddings = []
                for c in entities[target_entity]['candidates']:
                    if c != target_entity:
                        negative_candidate = entities[target_entity]['candidates'][c]
                        n_c_embedding = get_candidate_embedding(negative_candidate)
                        negative_candidate_embeddings.append(n_c_embedding)

                # Ranking Loss
                mention_loss = 0.0
                mention_loss += (1.0 - self.scorer(mention_embedding, p_c_embedding))
                for i, n_c_embedding in enumerate(negative_candidate_embeddings):
                    mention_loss += max(0, self.scorer(mention_embedding, n_c_embedding))
                el_loss.append(mention_loss)
            el_loss = torch.cat(el_loss, dim=0).mean() # ???

        elif mode == 'eval':
            predicted_ranks = []
            for k in mention_span_token_embeddings:
                # Mention Embedding by mean pooling the embeddings of the mention span tokens
                mention_embedding = torch.cat(mention_span_token_embeddings[k], dim=0).mean(dim=0).unsqueeze(0)

                target_entity = idx_to_entities[k]
                candidate_scores = {}
                for c in entities[target_entity]['candidates']:
                    candidate = entities[target_entity]['candidates'][c]
                    c_embedding = get_candidate_embedding(candidate)
                    candidate_scores[c] = self.scorer(mention_embedding, c_embedding)[0].item()

                # Rank the scores and get the ranking of the target candidate
                candidate_scores = sorted(candidate_scores.items(), key=lambda x: x[1])
                for i, (key, value) in enumerate(candidate_scores):
                    if key == entities[target_entity]['label']:
                        predicted_ranks.append(i+1)
            predicted_ranks = torch.LongTensor(predicted_ranks).cuda()

        # **********************  <<<<< Named Entity Recognition >>>>>> ************************
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # for NER
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # For NER loss (classification loss)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                ner_loss = loss_fct(active_logits, active_labels)
            else:
                ner_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Final outputs
        if mode == 'train':
            outputs = (ner_loss + el_loss,) + outputs
        elif mode == 'eval':
            outputs = (ner_loss,) + (predicted_ranks,) + outputs

        return outputs  # (loss), (predicted_ranks), ner_scores, (hidden_states), (attentions)