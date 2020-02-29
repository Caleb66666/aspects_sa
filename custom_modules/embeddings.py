# @File: transfer_embedding
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/5 16:53:25

import torch
from torch import nn
from custom_modules.fusions import Highway, SfuCombiner, BasicSfu, Concat
from torch.autograd import Variable


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


class TransferEmbeddingWithSqueezer(nn.Module):
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


class TransferEmbedding(nn.Module):
    def __init__(self, transfer_cls, transfer_path, embedding_attributes, fine_tune_embed=True):
        super().__init__()

        transfer_model = transfer_cls.from_pretrained(transfer_path)
        [setattr(p, "requires_grad", False) for p in transfer_model.parameters()]
        if isinstance(embedding_attributes, str):
            embedding_attributes = (embedding_attributes,)
        self.embedding = self.obtain_token_embedding(transfer_model, embedding_attributes)
        if fine_tune_embed:
            [setattr(p, "requires_grad", True) for p in self.embedding.parameters()]

    @staticmethod
    def obtain_token_embedding(transfer_model, embedding_attributes):
        embedding = transfer_model
        for attr in embedding_attributes:
            embedding = getattr(embedding, attr)
        return embedding

    def forward(self, seq_ids, seq_len):
        return self.embedding(seq_ids)


class WordCharEmbeddingWithCnn(nn.Module):
    def __init__(self, word_vocab_size, word_embed_size, char_vocab_size, char_embed_size, n_channel, kernel_sizes,
                 max_seq=1000, highway_layers=3, positional_encoding=False):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.max_seq = max_seq

        self.word_embed_size = word_embed_size
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_embed_size)

        self.char_embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, self.char_embed_size)
        self.char_vec_dim = n_channel * len(kernel_sizes)
        self.cnn_list = nn.ModuleList([nn.Conv2d(1, n_channel, [ks, self.char_embed_size]) for ks in kernel_sizes])

        self.embed_size = self.word_embed_size + self.char_vec_dim
        self.highway = Highway(self.embed_size, num_layers=highway_layers)

        if self.positional_encoding:
            self.positional_embed = self.make_positional_encoding(self.embed_size, max_seq=max_seq)

    @staticmethod
    def make_positional_encoding(embed_size, max_seq):
        pe = torch.arange(0, max_seq).unsqueeze(1).expand(max_seq, embed_size).contiguous()
        div_term = torch.pow(10000, torch.arange(0, embed_size * 2, 2) / embed_size)
        pe = (pe / div_term).float()
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe

    def load_pre_trained_embeddings(self, pre_word_embedding=None, pre_char_embedding=None):
        self.word_embedding.from_pretrained(torch.from_numpy(pre_word_embedding))
        self.char_embedding.from_pretrained(torch.from_numpy(pre_char_embedding))

    def obtain_word_embed(self, word_idx):
        return self.word_embedding(word_idx)

    def obtain_char_embed(self, char_idx):
        batch_size, max_seq, max_char_len = char_idx.size()
        char_idx = char_idx.view(-1, char_idx.size(2))
        raw_char_embed = self.char_embedding(char_idx).unsqueeze(1)
        char_embed = []
        for cnn_unit in self.cnn_list:
            cnn_raw_embed = cnn_unit(raw_char_embed).squeeze(-1)
            pooled_cnn = torch.max_pool1d(cnn_raw_embed, cnn_raw_embed.size(-1)).squeeze(-1)
            char_embed.append(pooled_cnn)
        char_embed = torch.cat(char_embed, dim=-1)
        char_embed = char_embed.view(batch_size, max_seq, -1)
        return char_embed

    def forward(self, word_idx, char_idx):
        char_idx = char_idx.view(word_idx.size(0), word_idx.size(1), -1)

        word_embed = self.obtain_word_embed(word_idx)
        char_embed = self.obtain_char_embed(char_idx)
        embedding = self.highway(torch.cat([word_embed, char_embed], dim=-1))

        if self.positional_encoding:
            # positional_vector = nn.Parameter(self.positional_embed[:self.max_seq], requires_grad=True)
            positional_vector = Variable(self.positional_embed[:self.max_seq].to(word_idx.device))
            embedding += positional_vector

        return embedding


class WordCharEmbeddingWithRnn(nn.Module):
    def __init__(self, word_vocab_size, word_embed_size, char_vocab_size, char_embed_size, rnn_hidden_size,
                 rnn_layers=1, bidirectional=True, positional_encoding=False, fusion_method="cat", max_seq=512):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.max_seq = max_seq

        self.word_embed_size = word_embed_size
        self.word_embedding = nn.Embedding(word_vocab_size, self.word_embed_size)

        self.char_embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, self.char_embed_size)
        self.char_bidirectional = bidirectional
        self.char_encoder = nn.LSTM(self.char_embed_size, rnn_hidden_size, num_layers=rnn_layers,
                                    bidirectional=self.char_bidirectional, batch_first=True)

        if bidirectional:
            char_hidden_size = 2 * rnn_hidden_size
        else:
            char_hidden_size = rnn_hidden_size

        if fusion_method.lower() == "cat":
            self.fusion_model = Concat()
            self.embed_size = self.word_embed_size + char_hidden_size
        elif fusion_method.lower() == "sfu":
            self.fusion_model = BasicSfu(self.word_embed_size, char_hidden_size)
            self.embed_size = self.word_embed_size
        else:
            raise TypeError(f"bad fusion method: {fusion_method}")

        if self.positional_encoding:
            self.positional_embed = self.make_positional_encoding(self.embed_size, max_seq=self.max_seq)

    @staticmethod
    def make_positional_encoding(embed_size, max_seq):
        pe = torch.arange(0, max_seq).unsqueeze(1).expand(max_seq, embed_size).contiguous()
        div_term = torch.pow(10000, torch.arange(0, embed_size * 2, 2) / embed_size)
        pe = (pe / div_term).float()
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe

    def obtain_word_embed(self, word_idx):
        return self.word_embedding(word_idx)

    @staticmethod
    def char_pooling(tensor):
        max_p = torch.max_pool1d(tensor.transpose(1, 2), tensor.size(1)).squeeze(-1)
        return max_p

    def obtain_char_embed(self, char_idx):
        batch_size, max_seq, max_char_len = char_idx.size()
        char_idx = char_idx.view(-1, char_idx.size(2))
        raw_char_embed = self.char_embedding(char_idx)  # batch * max_seq, max_char_len, char_dim
        char_hidden, _ = self.char_encoder(raw_char_embed)
        char_pooled = self.char_pooling(char_hidden)
        return char_pooled.view(batch_size, max_seq, -1)

    def forward(self, word_idx, char_idx):
        char_idx = char_idx.view(word_idx.size(0), word_idx.size(1), -1)

        word_embed = self.obtain_word_embed(word_idx)  # batch_size, max_seq, word_embed_size
        char_embed = self.obtain_char_embed(char_idx)  # batch_size, max_seq, 2 * char_hidden_size
        embedding = self.fusion_model(word_embed, char_embed)
        if self.positional_encoding:
            # positional_vector = nn.Parameter(self.positional_embed[:self.max_seq], requires_grad=True)
            positional_vector = Variable(self.positional_embed[:self.max_seq])
            embedding += positional_vector
        return embedding


if __name__ == '__main__':
    pass
