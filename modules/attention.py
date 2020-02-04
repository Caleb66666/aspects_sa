# @File: attention
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/4 16:50:41

import torch
from torch import nn
from modules.utility import masked_softmax, activations
from modules.similarity import DotProductSimilarity


class BaseAttention(nn.Module):
    """
    An `Attention` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.

    Inputs:
    - vector: shape `(batch_size, embedding_dim)`
    - matrix: shape `(batch_size, num_rows, embedding_dim)`
    - matrix_mask: shape `(batch_size, num_rows)`, specifying which rows are just padding.

    Output:
    - attention: shape `(batch_size, num_rows)`.

    # Parameters
    normalize : `bool`, optional (default : `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, normalize=True):
        super(BaseAttention, self).__init__()
        self._normalize = normalize

    def forward(self, vector, matrix, matrix_mask=None):
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        return similarities

    def _forward_internal(self, vector, matrix):
        raise NotImplementedError


class LegacyAttention(BaseAttention):
    def __init__(self, similarity_fn=None, normalize=True):
        super(LegacyAttention, self).__init__(normalize)

        self._similarity_fn = similarity_fn or DotProductSimilarity()

    def _forward_internal(self, vector, matrix):
        tiled_vector = vector.unsqueeze(1).expand(vector.size(0), matrix.size(1), vector.size(1))
        return self._similarity_fn(tiled_vector, matrix)


class AdditiveAttention(BaseAttention):
    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self._w_matrix = nn.Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = nn.Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = nn.Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)


class BiLinearAttention(BaseAttention):
    def __init__(self, vector_dim, matrix_dim, activation, normalize):
        super().__init__(normalize)
        self._weight_matrix = nn.Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = nn.Parameter(torch.Tensor(1))
        self._activation = activation or activations.get("linear")()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        return self._activation(intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias)
