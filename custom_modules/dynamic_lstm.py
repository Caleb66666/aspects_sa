# @File: dynamic_lstm
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/1 17:34:57

import torch
from torch import nn


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=True, sort_within_batch=False):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sort_within_batch = sort_within_batch

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def forward(self, x, x_len):
        if self.sort_within_batch:
            p_x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
            output, (ht, ct) = self.rnn(p_x, None)
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)[0]
            return output, (ht, ct)
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x, x_len = x[x_sort_idx], x_len[x_sort_idx]
        p_x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        output, (ht, ct) = self.rnn(p_x, None)
        ht = ht.transpose(0, 1)[x_unsort_idx].transpose(0, 1)
        ct = ct.transpose(0, 1)[x_unsort_idx].transpose(0, 1)
        out = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)[0][x_unsort_idx]
        return out, (ht, ct)


if __name__ == '__main__':
    from utils.ml_util import gen_random_mask

    num_layers_ = 3
    batch_size_ = 5
    embed_dim_ = 7
    hidden_size_ = 11
    max_seq_ = 13
    bidirectional_ = True

    embed_seq_ = torch.randn(size=[batch_size_, max_seq_, embed_dim_])
    seq_mask_, seq_len_ = gen_random_mask(batch_size_, max_seq_)

    encoder = DynamicLSTM(embed_dim_, hidden_size_, num_layers_, bidirectional=bidirectional_)
    encoder(embed_seq_, seq_len_)
