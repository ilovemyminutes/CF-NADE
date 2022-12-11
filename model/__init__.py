import math

from torch import nn


def initialize(weight: nn.Parameter, bias: nn.Parameter = None):
    """
    :param weight:
    :param bias:
    :return:
    """
    # initialize W & c
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
