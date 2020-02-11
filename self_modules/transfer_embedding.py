# @File: transfer_embedding
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/5 16:53:25

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


class TransferEmbedding(nn.Module):
    def __init__(self, transfer_cls, transfer_path, embedding_attributes, fine_tune_embed=True):
        super().__init__()

        self.squeezer = Squeezer()
        transfer_model = transfer_cls.from_pretrained(transfer_path)
        [setattr(p, "requires_grad", False) for p in transfer_model.parameters()]
        if isinstance(embedding_attributes, str):
            embedding_attributes = (embedding_attributes,)
        self.embedding = self.obtain_word_embedding(transfer_model, embedding_attributes)
        if fine_tune_embed:
            [setattr(p, "requires_grad", True) for p in self.embedding.parameters()]

    @staticmethod
    def obtain_word_embedding(transfer_model, embedding_attributes):
        embedding = transfer_model
        for attr in embedding_attributes:
            embedding = getattr(embedding, attr)
        return embedding

    def forward(self, seq_ids, seq_len):
        embed_seq = self.embedding(seq_ids)
        return self.squeezer(embed_seq, seq_len)
