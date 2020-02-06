# @File: squeeze_embedding
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/1 16:24:54

import torch
from torch import nn


class Squeezer(nn.Module):
    def __init__(self, batch_first=True):
        super(Squeezer, self).__init__()

        self.batch_first = batch_first

    def forward(self, x, x_len):
        # 从大到小排序
        x_sort_idx = torch.sort(-x_len)[1].long()
        # 经过从大到小排序后，记录原来的索引位置
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()

        # 重新排列
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]

        # pack
        p_x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # unpack
        up_x = nn.utils.rnn.pad_packed_sequence(p_x, batch_first=self.batch_first)

        out = up_x[0]
        return out[x_unsort_idx]


if __name__ == '__main__':
    from utils.ml_util import gen_random_mask

    batch_size_ = 4
    embed_dim_ = 5
    max_seq_ = 19

    embed_seq_ = torch.randn(size=[batch_size_, max_seq_, embed_dim_])
    seq_mask_, seq_len_ = gen_random_mask(batch_size_, max_seq_, 3)
    print(seq_mask_)

    squeezer = Squeezer()
    res = squeezer(embed_seq_, seq_len_)
    print(res.size())
