# @File: attention
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/1 19:20:55

import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, output_dim=None, n_head=1, score_fn='dot_product', dropout=0):
        super(Attention, self).__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if output_dim is None:
            output_dim = embed_dim

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_head = n_head
        self.score_fn = score_fn

        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.projection = nn.Linear(n_head * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if self.score_fn == "mlp":
            self.weight = nn.Parameter(torch.zeros(size=(hidden_dim * 2)), requires_grad=True)
        elif self.score_fn == "bi_linear":
            self.weight = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)), requires_grad=True)
        else:  # dot_product / scaled_dot_product
            self.weight = None

        self.reset_params()

    def reset_params(self):
        sd = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-sd, sd)

    def forward(self, k, q):
        if len(k.shape) == 2:
            k = torch.unsqueeze(k, dim=1)
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        assert k.shape[0] == q.shape[0], "key_batch_size doesn't match query_batch_size"
        mb_size = k.shape[0]  # mini-batch-size
        k_len, q_len = k.shape[1], q.shape[1]

        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_fn == "dot_product":
            score = torch.bmm(qx, kx.permute(0, 2, 1))
        elif self.score_fn == "scaled_dot_product":
            score = torch.bmm(qx, kx.permute(0, 2, 1))
            score = torch.div(score, math.sqrt(self.hidden_dim))
        elif self.score_fn == "mlp":
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_fn == "bi_linear":
            qw = torch.matmul(qx, self.weight)
            score = torch.bmm(qw, kx.permute(0, 2, 1))
        else:
            raise RuntimeError(f"invalid score function: {self.score_fn}")

        score = torch.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # # (?, q_len, n_head*hidden_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    def __init__(self, embed_dim, hidden_dim=None, output_dim=None, n_head=1, score_fn="dot_product", dropout=0,
                 q_len=1):
        super().__init__(embed_dim, hidden_dim, output_dim, n_head, score_fn, dropout)

        self.q_len = q_len
        self.q = nn.Parameter(torch.zeros(size=(q_len, embed_dim)), requires_grad=True)
        self.reset_q()

    def reset_q(self):
        sd = 1.0 / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-sd, sd)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diagonal=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diagonal = diagonal

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        # Project vectors
        if self.linear:
            x_projection = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_projection = torch.relu(x_projection)
        else:
            x_projection = x

        # Compute scores
        scores = x_projection.bmm(x_projection.transpose(2, 1))
        if not self.diagonal:
            x_len = x.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0

        # Mask padding
        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores = torch.masked_fill(scores, x_mask.to(dtype=torch.bool), -float('inf'))

        # Normalize with softmax
        alpha = torch.softmax(scores, dim=2)

        # Take weighted average
        matched_seq = alpha.bmm(x)
        return matched_seq


if __name__ == '__main__':
    from utils.ml_util import gen_random_mask

    num_layers_ = 3
    batch_size_ = 5
    embed_dim_ = 7
    hidden_size_ = 11
    max_seq_ = 13
    bidirectional_ = True
    n_head_ = 2
    num_classes_ = 4

    embed_seq_ = torch.randn(size=[batch_size_, max_seq_, embed_dim_])
    seq_mask_, seq_len_ = gen_random_mask(batch_size_, max_seq_)

    # attention_ = Attention(embed_dim_, hidden_size_, num_classes_, n_head_, )
    self_attn = SelfAttnMatch(embed_dim_)
    print(self_attn(embed_seq_, seq_mask_).size())
