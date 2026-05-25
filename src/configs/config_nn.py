# src/configs/config_nn.py
"""
Default network configurations.

Each mapping describes the *kwargs* passed by
`src.nn.generator.create_network_from_config` to the corresponding network
class. User configs (e.g. inside `EncoderCfg.net` or `NSDECfg.drift`) override
the relevant subset; missing keys fall back to these defaults.

Required key
------------
- ``type``: one of {"mlp", "rnn", "lstm", "gru", "mamba", "constant", "affine"}.
"""

DEFAULT_CONFIG_MLP = {
    'type': 'mlp',
    'n_layers': 3,
    'n_units': 64,
    'dropout': None,
    'activation': 'ReLU',
    'out_activation': 'Identity',
}

DEFAULT_CONFIG_RNN = {
    'type': 'rnn',
    'n_layers': 2,
    'n_units': 128,
    'dropout': 0.1,
    'out_activation': 'Identity',
    'bidirectional': False,
}

DEFAULT_CONFIG_LSTM = {
    'type': 'lstm',
    'n_layers': 2,
    'n_units': 128,
    'dropout': 0.1,
    'out_activation': 'Identity',
    'bidirectional': False,
}

DEFAULT_CONFIG_GRU = {
    'type': 'gru',
    'n_layers': 2,
    'n_units': 128,
    'dropout': 0.1,
    'out_activation': 'Identity',
    'bidirectional': False,
}

DEFAULT_CONFIG_CONSTANT = {
    "type": "constant",
    "out_activation": 'Identity',
    "init": "normal",       # or "zeros"
    "init_std": 0.05,
}

DEFAULT_CONFIG_AFFINE = {
    "type": "affine",
    "bias": True,
    "out_activation": 'Identity',
}

DEFAULT_CONFIG_MAMBA = {
    'type': 'mamba',
    'n_layers': 2,
    'n_units': 64,
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
    'dropout': 0.1,
    'out_activation': 'Identity',
}
