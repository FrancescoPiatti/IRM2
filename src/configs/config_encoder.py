# src/configs/config_encoder.py
from dataclasses import dataclass

from typing import Any
from typing import Mapping
from typing import Literal
from typing import Optional

import warnings

from ..utils.misc import freeze_dict


EncoderMode = Literal["simple", "hierarchical"]
# Input preprocessing applied by ShortRateModel before the encoder:
# - 'none'     : raw decimal rates (~0.00-0.06) — day-to-day differences are
#                ~5e-4, a very weak signal for the encoder.
# - 'scale100' : multiply by 100 (back to percent units). Preserves level,
#                slope and curvature exactly; just rescales so the encoder
#                sees O(1) inputs with O(0.05) daily moves. RECOMMENDED.
# - 'norm_z'   : per-feature z-score over the lookback window. NOTE: this
#                removes each pillar's window mean, destroying level
#                information — only use if you want shape-only encoding.
# - 'norm_max' : divide by the max |value| over the window.
PreprocessMode = Literal['none', 'scale100', 'norm_z', 'norm_max']
CombineMethod = Literal['concat', 'project', 'add']
OutNorm = Literal['layernorm', 'rmsnorm', 'none', 'None']


# Change here to a better one
def _default_encoder_net() -> Mapping[str, Any]:
    return freeze_dict({'type': 'lstm'})


@dataclass
class EncoderCfg:
    """
    Encoder configuration.

    The config is mode-driven:
    - mode="simple" uses `net` only
    - mode="hierarchical" uses `fast_net`, `slow_net`, and `combine`

    Attributes
    ----------
    mode : Literal["simple","hierarchical"]
        Selects the encoder topology.
    net : Optional[Mapping[str, Any]]
        Backbone config for simple mode.
    fast_net : Optional[Mapping[str, Any]]
        Fast stream backbone config for hierarchical mode.
    slow_net : Optional[Mapping[str, Any]]
        Slow stream backbone config for hierarchical mode.
    combine : CombineMethod
        How to combine fast/slow embeddings in hierarchical mode.
    """
    # Mode
    mode: EncoderMode = "simple"

    # Common options
    preprocess_mode: Optional[PreprocessMode] = None
    out_norm: Optional[OutNorm] = "layernorm"

    # Simple
    net: Optional[Mapping[str, Any]] = None

    # Hierarchical
    fast_net: Optional[Mapping[str, Any]] = None
    slow_net: Optional[Mapping[str, Any]] = None
    combine: CombineMethod = "concat"


    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> None:
        """
        Validate and normalise the encoder config.

        Call after all fields are set. This method:
        - Lowercases mode
        - Fills defaults for missing network configs
        - Nullifies fields that don't apply to the selected mode (with warnings)
        - Validates that network specs contain required keys
        """
        self.mode = str(self.mode).lower()

        if self.mode not in ("simple", "hierarchical"):
            raise ValueError(f"Unknown EncoderCfg.mode='{self.mode}'. Expected 'simple' or 'hierarchical'.")

        if self.out_norm not in (None, "none", "None", "layernorm", "rmsnorm"):
            raise ValueError(f"Invalid EncoderCfg.out_norm='{self.out_norm}'.")

        if self.preprocess_mode not in (None, 'none', 'scale100', 'norm_z', 'norm_max'):
            raise ValueError(f"Invalid EncoderCfg.preprocess_mode='{self.preprocess_mode}'.")

        if self.mode == "simple":
            # Fill defaults
            if self.net is None:
                self.net = _default_encoder_net()

            # Conflicts: hierarchical fields provided -> ignore with warning
            if self.fast_net is not None or self.slow_net is not None:
                warnings.warn(
                    "EncoderCfg(mode='simple'): fast_net/slow_net were provided but will be ignored.",
                    category=UserWarning,
                    stacklevel=2,
                )
            # Remove hierarchical-only fields. Note that `combine` is reset
            # to None for simple mode — readers who introspect cfg.combine
            # after validate() must accept None for the simple variant.
            self.fast_net = None
            self.slow_net = None
            self.combine = None

            # Validate network spec
            self._check_net_spec(self.net, "cfg.net")

        elif self.mode == "hierarchical":
            # Fill defaults
            if self.fast_net is None:
                self.fast_net = _default_encoder_net()
            if self.slow_net is None:
                self.slow_net = _default_encoder_net()

            # Conflicts: simple fields provided -> ignore with warning
            if self.net is not None:
                warnings.warn(
                    "EncoderCfg(mode='hierarchical'): net was provided but will be ignored.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # Remove simple-only fields
            self.net = None

            # Validate combine method
            if str(self.combine).lower() not in ("concat", "add", "project"):
                raise ValueError(f"Invalid EncoderCfg.combine='{self.combine}'.")

            # Validate network specs
            self._check_net_spec(self.fast_net, "cfg.fast_net")
            self._check_net_spec(self.slow_net, "cfg.slow_net")


    @staticmethod
    def _check_net_spec(spec: Mapping[str, Any], name: str) -> None:
        """
        Validate that a network spec is a mapping with a supported 'type' key.
        """
        if "type" not in spec:
            raise ValueError(f"{name} must contain key 'type' (e.g. {{'type': 'lstm'}}).")
        if spec["type"] not in ("rnn", "gru", "lstm", "mamba"):
            raise ValueError(f"Unsupported network type '{spec['type']}' in {name}. (Encoder does not support MLPs.)")
