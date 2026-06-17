
# src/nn/activations.py
import torch.nn as nn

ACTIVATION_REGISTRY = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
    'id': nn.Identity,
    'softmax': lambda: nn.Softmax(dim=1),
    'softplus': nn.Softplus,
    'gelu': nn.GELU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    # Add more activations as needed
}


def _get_activation(activation):
    """
    Get a fresh activation function instance.
    Supported types are:
    - str: name of the activation function (returns a new instance each time)
    - nn.Module: an instance of a PyTorch activation function (returned as-is)
    - type (class): a callable nn.Module subclass (invoked to create an instance)
    - list: a list of activation functions (either str or nn.Module)
    """

    if isinstance(activation, str):
        s = activation.strip()
        # Recover a serialized class repr, e.g. when a config default that
        # was the *class* ``nn.Identity`` got JSON-dumped via ``str(cls)``
        # into "<class 'torch.nn.modules.linear.Identity'>". Round-trips
        # through model_info / config serialization land here.
        if s.startswith("<class") and "." in s:
            s = s.rstrip("'>").split(".")[-1]
        key = s.lower()
        if key in ACTIVATION_REGISTRY:
            return ACTIVATION_REGISTRY[key]()
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    if isinstance(activation, list):
        return [_get_activation(act) for act in activation]

    if activation is None:
        return nn.Identity()

    else:
        raise TypeError(f"Activation must be a string or an instance of nn.Module. Input: {activation}")
