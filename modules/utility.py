# @File: utility
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/2/4 17:00:05

import torch


class Activation(object):
    def __init__(self):
        self._activations = dict({
            "linear": (lambda: lambda x: x, None),  # type: ignore
            "mish": (lambda: lambda x: x * torch.tanh(torch.nn.functional.softplus(x)), None),  # type: ignore
            "swish": (lambda: lambda x: x * torch.sigmoid(x), None),  # type: ignore
            "relu": (torch.nn.ReLU, None),
            "relu6": (torch.nn.ReLU6, None),
            "elu": (torch.nn.ELU, None),
            "prelu": (torch.nn.PReLU, None),
            "leaky_relu": (torch.nn.LeakyReLU, None),
            "threshold": (torch.nn.Threshold, None),
            "hardtanh": (torch.nn.Hardtanh, None),
            "sigmoid": (torch.nn.Sigmoid, None),
            "tanh": (torch.nn.Tanh, None),
            "log_sigmoid": (torch.nn.LogSigmoid, None),
            "softplus": (torch.nn.Softplus, None),
            "softshrink": (torch.nn.Softshrink, None),
            "softsign": (torch.nn.Softsign, None),
            "tanhshrink": (torch.nn.Tanhshrink, None),
            "selu": (torch.nn.SELU, None),
        })

    def get(self, key):
        return self._activations.get(key)[0]


activations = Activation()


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        memory_efficient: bool = True,
        mask_fill_value: float = -1e32,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broad-cast-able to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    un-squeeze on dimension 1 until they match.  If you need a different un-squeezing of your mask,
    do it yourself before passing the mask into this function.
    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        return torch.softmax(vector, dim=dim)

    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)

    if not memory_efficient:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.softmax(vector * mask, dim=dim)
        result = result * mask
        result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
    else:
        masked_vector = vector.masked_fill(
            (torch.tensor(1).to(mask.device) - mask).to(dtype=torch.bool), mask_fill_value
        )
        result = torch.softmax(masked_vector, dim=dim)
    return result


def _get_combination_dim(combination, tensor_dims) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise RuntimeError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise RuntimeError('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim


def get_combined_dim(combination, tensor_dims):
    combination = combination.replace("x", "1").replace("y", "2")
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(",")])


def combine_tensors(combination, tensors) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    `combination` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like `"1,2,1+2,3-1"`.
    We allow the following kinds of combinations : `x`, `x*y`, `x+y`, `x-y`, and `x/y`,
    where `x` and `y` are positive integers less than or equal to `len(tensors)`.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the `combination` string.  For example, for the input string `"1,2,1*2"`, the result
    would be `[1;2;1*2]`, as you would expect, where `[;]` is concatenation along the last
    dimension.
    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like `torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])`.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.
    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.
    This function also accepts `x` and `y` in place of `1` and `2` in the combination
    string.
    """
    if len(tensors) > 9:
        raise RuntimeError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(",")]
    return torch.cat(to_concatenate, dim=-1)


def _get_combination(combination, tensors) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise RuntimeError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise RuntimeError("Invalid operation: " + operation)
