# @File: similarity
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/4 17:18:04

import torch
from torch import nn
from modules.utility import activations
import math
from allennlp.nn.util import get_combined_dim, combine_tensors


class BaseSimilarity(nn.Module):
    default_implementation = "dot_product"

    def __init__(self):
        super(BaseSimilarity, self).__init__()

    def forward(self, tensor1, tensor2):
        """
        Takes two tensors of the same shape, such as `(batch_size, length_1, length_2,
        embedding_dim)`.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as `(batch_size, length_1, length_2)`.
        """
        raise NotImplementedError


class BiLinearSimilarity(BaseSimilarity):
    """
        This similarity function performs a bi-linear transformation of the two input vectors.  This
        function has a matrix of weights `W` and a bias `b`, and the similarity between two vectors
        `x` and `y` is computed as `x^T W y + b`.
        # Parameters
        tensor_1_dim : `int`
            The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
            length of the vector that will go into the similarity computation.  We need this so we can
            build the weight matrix correctly.
        tensor_2_dim : `int`
            The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
            length of the vector that will go into the similarity computation.  We need this so we can
            build the weight matrix correctly.
        activation : `Activation`, optional (default=linear (i.e. no activation))
            An activation function applied after the `x^T W y + b` calculation.  Default is no
            activation.
        """

    def __init__(self, tensor1_dim, tensor2_dim, activation=None):
        super().__init__()
        self._weight_matrix = nn.Parameter(torch.zeros(size=(tensor1_dim, tensor2_dim)), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(0).float(), requires_grad=True)
        self._activation = activation or activations.get("linear")()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, tensor1, tensor2):
        # 此处两个维度的weight即为所谓的bi-linear，对tensor1和tensor2作双向的transformer
        intermediate = torch.matmul(tensor1, self._weight_matrix)
        result = (intermediate * tensor2).sum(dim=-1)
        return self._activation(result + self._bias)


class CosineSimilarity(BaseSimilarity):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        normed_tensor1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
        normed_tensor2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
        return (normed_tensor1 * normed_tensor2).sum(dim=-1)


class DotProductSimilarity(BaseSimilarity):
    def __init__(self, scale_output=False):
        super().__init__()
        self.scale_output = scale_output

    def forward(self, tensor1, tensor2):
        result = (tensor1 * tensor2).sum(dim=-1)
        if self.scale_output:
            result *= math.sqrt(tensor1.size(-1))
        return result


class LinearSimilarity(BaseSimilarity):
    """
     This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.
    If the two vectors are `x` and `y`, we allow the following kinds of combinations : `x`,
    `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give `x,y,x*y` as the `combination` parameter to this class.  The computed similarity
    function would then be `w^T [x; y; x*y] + b`, where `w` is a vector of weights, `b` is a
    bias parameter, and `[;]` is vector concatenation.
    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.
    # Parameters
    tensor_1_dim : `int`
        The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : `int`
        The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : `str`, optional (default="x,y")
        Described above.
    activation : `Activation`, optional (default=linear (i.e. no activation))
        An activation function applied after the `w^T * [x;y] + b` calculation.  Default is no
        activation.
    """

    def __init__(self, tensor1_dim, tensor2_dim, combination="x,y", activation=None):
        super().__init__()

        self._combination = combination
        combined_dim = get_combined_dim(combination, [tensor1_dim, tensor2_dim])
        self._weight_vector = nn.Parameter(torch.zeros(size=[combined_dim]), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(0).float(), requires_grad=True)
        self._activation = activation or activations.get("linear")()
        self.reset_params()

    def reset_params(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def forward(self, tensor1, tensor2):
        combined_tensors = combine_tensors(self._combination, [tensor1, tensor2])
        dot_product = torch.matmul(combined_tensors, self._weight_vector)
        return self._activation(dot_product + self._bias)


if __name__ == '__main__':
    batch_size_ = 3
    embed_dim_ = 5
    tensor1_dim_ = 7
    tensor2_dim_ = 11

    # tensor1_ = torch.randn(size=(batch_size_, tensor1_dim_))
    # tensor2_ = torch.randn(size=(batch_size_, tensor2_dim_))
    # similar = BiLinearSimilarity(tensor1_dim_, tensor2_dim_)
    # print(similar(tensor1_, tensor2_))

    # tensor1_ = torch.randn(size=(batch_size_, tensor1_dim_))
    # tensor2_ = torch.randn(size=(batch_size_, tensor1_dim_))
    # similar = CosineSimilarity()
    # print(similar(tensor1_, tensor2_))

    # tensor1_ = torch.randn(size=(batch_size_, tensor1_dim_))
    # tensor2_ = torch.randn(size=(batch_size_, tensor1_dim_))
    # similar = DotProductSimilarity()
    # print(similar(tensor1_, tensor2_))

    tensor1_ = torch.randn(size=(batch_size_, tensor1_dim_))
    tensor2_ = torch.randn(size=(batch_size_, tensor2_dim_))
    similar = LinearSimilarity(tensor1_dim_, tensor2_dim_)
    print(similar(tensor1_, tensor2_))
