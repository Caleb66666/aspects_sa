# @File: transfer_embedding
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/5 16:53:25

from self_modules.squeeze_embedding import Squeezer
from torch import nn


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
