import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEmbedding(nn.Module):
    def __init__(self, config, utils, device):
        super(CharEmbedding, self).__init__()
        self.device = device
        self.char_emb = nn.Embedding(utils.num_char, config.emb_dim)
        self.conv1 = nn.Conv1d(in_channels=config.emb_dim, out_channels=config.out_dim, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=config.emb_dim, out_channels=config.out_dim, kernel_size=2)
        # self.conv3 = nn.Conv1d(in_channels=config.emb_dim, out_channels=config.out_dim, kernel_size=3)
        self.linear = nn.Linear(config.out_dim, config.out_dim)

    def forward(self, char_seq):
        char_seq = torch.LongTensor([char_seq]).to(self.device)
        char_embs = self.char_emb(char_seq)
        char_embs = char_embs.transpose(1, 2)  # batch_size X in_channels X seq_len
        out1 = self.conv1(char_embs)
        out1 = torch.mean(out1, dim=2)  # Mean pooling
        # out2 = self.conv2(char_embs)
        # out2 = torch.mean(out2, dim=2)  # Mean pooling
        # out3 = self.conv3(char_embs)
        # out3 = torch.mean(out3, dim=2)  # Mean pooling

        # cnn_output = torch.cat([out1, out2, out3], dim=1)
        output_emb = self.linear(out1)
        return output_emb


class EmbeddingLayer(nn.Module):
    def __init__(self, config, utils, device):
        super().__init__()
        self.config = config
        self.utils = utils
        self.device = device
        self.char_emb = CharEmbedding(config, utils, device)

    def forward(self, word_tokens):
        batch_size = word_tokens.size(0)
        batch_word_embeddings = []
        for i in range(batch_size):
            word_embeddings = []
            for w_t in word_tokens[i]:
                w_t = w_t.cpu().item()
                w = self.utils.convert_index_to_word(w_t)
                w_e = self.utils.get_glove_embeddings(w) # GloVe word embedding
                w_e = torch.FloatTensor(w_e).to(self.device)
                char_ids = self.utils.convert_word_to_char_idx(w)
                w_c = self.char_emb(char_ids) # Character embedding
                word_embeddings.append(torch.cat([w_e, w_c], dim=1))
            word_embeddings = torch.cat(word_embeddings, dim=0)
            batch_word_embeddings.append(word_embeddings.unsqueeze(0))

        batch_word_embeddings = torch.cat(batch_word_embeddings, dim=0)
        return batch_word_embeddings


class Encoder(nn.Module):
    def __init__(self, config, utils):
        super().__init__()
        self.biLSTM = nn.LSTM(input_size=config.input_dim,
                              hidden_size=config.hidden_dim,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)

    def forward(self, input_seq):
        # Equation 1. of the paper
        self.biLSTM.flatten_parameters()
        output, (h, c) = self.biLSTM(input_seq)
        return output

class CrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_a = nn.Linear(3*config.hidden_dim*2, 1, bias=False)
        self.ffn = nn.Linear(4*config.hidden_dim*2*config.candidate_len, 1)

    def forward(self, u_p, u_c):
        '''
        :param u_p: encoded mention with context
        :param u_c: encoded candidates
        :return: attention-based relevance score f
        '''
        m = u_p.size(0)
        n = u_c.size(0)

        S = []
        for i in range(m):
            for j in range(n):
                s_ij = self.w_a(torch.cat([u_p[i], u_c[j], u_p[i] * u_c[j]], dim=0))
                s_ij = s_ij.reshape(-1)
                S.append(s_ij)
        S = torch.cat(S, dim=0)
        S = S.reshape(m, n)

        # Mention-to-candiate attention
        S_alpha = F.softmax(S, dim=1)   # Row wise softmax mention_len X candidate_len
        # Candidate-to-mention attention
        S_beta = F.softmax(S.t(), dim=1)  # Column wise softmax candidate_len X mention_len


        X = []
        for k in range(n):
            # Attention over candidate tokens
            a_k_alpha = 0
            for j in range(n):
                a_k_alpha += S_alpha[k][j] * u_c[j]

            # Attention over mention tokens
            a_k_beta = 0
            for i in range(m):
                a_k_beta += S_beta[k][i] * u_p[i]

            x = []
            x.append(u_p[k])
            x.append(a_k_alpha)
            x.append(u_p[k] * a_k_alpha)
            x.append(u_c[k] * a_k_beta)
            x = torch.cat(x, dim=0)
            X.append(x)

        X = torch.cat(X, dim=0)
        f = F.relu(self.ffn(X))

        return f


class LatentTypeSimilarity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_p = nn.Linear(config.mention_len*config.hidden_dim*2, config.num_latent_types)
        self.linear_c = nn.Linear(config.candidate_len*config.hidden_dim*2, config.num_latent_types)

    def forward(self, mention_embs, candidates_embs):
        b_size = mention_embs.size(0)
        mention_embs = mention_embs.reshape(b_size, -1)
        candidates_embs = candidates_embs.reshape(b_size, -1)

        # Equation 6. of the paper
        v_p = self.linear_p(mention_embs)
        v_p_ = F.softmax(v_p, dim=1)

        v_c = self.linear_c(candidates_embs)
        v_c_ = F.softmax(v_c, dim=1)

        g = F.cosine_similarity(v_p_, v_c_, dim=1)
        return v_p, v_c, g


class KnownTypeClassifier(nn.Module):
    def __init__(self, config, utils):
        super().__init__()
        self.linear_kt = nn.Linear(config.num_latent_types, utils.num_types)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, labels):
        # Equation 7. of the paper
        output = self.linear_kt(input)
        output = F.relu(output)
        # Equation 9. of the paper
        loss = self.criterion(output, labels)
        return loss


class RankingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_f = nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.w_g = nn.Parameter(torch.randn(1, 1), requires_grad=True)

    def forward(self, f, g):
        r = torch.mm(f, self.w_f.data) + torch.mm(g, self.w_g.data)
        return r


class LATTE(nn.Module):
    def __init__(self, config, utils, device):
        super().__init__()
        self.emb = EmbeddingLayer(config, utils, device)
        self.encoder = Encoder(config, utils)
        self.cross_attn = CrossAttentionLayer(config)
        self.lt_similarity = LatentTypeSimilarity(config)
        self.kt_classifier = KnownTypeClassifier(config, utils)
        self.ranking_layer = RankingLayer(config)
        self.init_weights()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, (nn.Linear)) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, mention, candidates, target_candidate, mention_type, candidate_types):
        '''
        :param mention: batch_size X mention_seq_len
        :param candidates: batch_size X num_candidates X candidate_seq_len
        :param target_candidate: batch_size X 1
        :param mention_type: batch_size X 1
        :param candidate_types: batch_size X num_candidates X 1
        :return: loss, scores
        '''
        # Mention Encoding
        mention_embs = self.emb(mention)  # batch_size X mention_seq_len X emb_dim
        encoded_mention = self.encoder(mention_embs)  # batch_size X mention_seq_len X hidden_dim*2
        b_size, m_seq_len, emb_dim = encoded_mention.size()
        b_size, num_cand, cand_seq_len = candidates.size()
        encoded_mention = encoded_mention.unsqueeze(1)  # batch_size X 1 X mention_seq_len X hidden_dim*2
        encoded_mention = encoded_mention.expand(-1, num_cand, -1, -1)  # batch_size X num_candidates X mention_seq_len X hidden_dim*2
        encoded_mention = encoded_mention.reshape(-1, m_seq_len, emb_dim) # (batch_size * num_candidates) X mention_seq_len X hidden_dim*2

        # Candidate Encoding
        candidates = candidates.reshape(-1,  cand_seq_len)  # (batch_size * num_candidates) X candidate_seq_len
        candidates_embs = self.emb(candidates)  # (batch_size * num_candidates) X candidate_seq_len X emb_dim
        encoded_candidates = self.encoder(candidates_embs)  # (batch_size * num_candidates) X candidate_seq_len X hidden_dim*2

        # Cross Attention
        f = []
        for i in range(encoded_mention.size(0)):
            f_i = self.cross_attn(encoded_mention[i], encoded_candidates[i])
            f.append(f_i)
        f = torch.cat(f, dim=0)

        # Latent type logits and similarity scores
        v_p, v_c, g = self.lt_similarity(encoded_mention, encoded_candidates)

        # Known type classifier
        v_p = v_p.reshape(b_size, num_cand, -1)
        v_p = v_p[:, 0, :]  # batch_size X num_latent_type
        classifier_input = torch.cat([v_p, v_c], dim=0)
        targets = torch.cat([mention_type.reshape(-1), candidate_types.reshape(-1)], dim=0)
        classification_loss = self.kt_classifier(classifier_input, targets)

        # Ranking
        f = f.reshape(-1, 1)
        g = g.reshape(-1, 1)

        scores = self.ranking_layer(f, g)
        scores = scores.reshape(b_size, -1)
        _, num_candiates = scores.size()

        # Ranking Loss
        ranking_loss = 0
        for i in range(b_size):
            r_loss = 0.0
            pos_score = scores[i][0]  # positive candidate at 0-th index always
            for j in range(1, num_candiates):
                neg_score = scores[i][j]
                r_loss += max(0, 1.0 - pos_score + neg_score)  # Pairwise ranking loss
            r_loss = r_loss / num_candiates
            ranking_loss += r_loss

        ranking_loss = ranking_loss / b_size

        loss = ranking_loss + classification_loss

        return loss, scores
