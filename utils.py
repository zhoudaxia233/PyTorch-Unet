import torch
import torch.nn as nn


def get_resnet_output_size(model, n, input_shape=(1, 3, 224, 224)):
    """Get the output size of a model's first n layers
    """
    dummy_input = torch.rand(input_shape)
    first_n_layers = list(model.children())[:n]
    m = nn.Sequential(*first_n_layers)
    return m(dummy_input).size()


def get_vgg_output_size(model, n, input_shape=(1, 3, 224, 224)):
    """Get the output size of a model's first n layers
    """
    dummy_input = torch.rand(input_shape)
    first_n_layers = model.features[:n]
    m = nn.Sequential(*first_n_layers)
    return m(dummy_input).size()
