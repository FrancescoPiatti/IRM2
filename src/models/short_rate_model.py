# src/models/short_rate_model.py
import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional
from typing import Union
from typing import Dict
from typing import Any
from typing import Mapping
from typing import Tuple

import os
from datetime import datetime
from pathlib import Path

from ..nn.generator import create_network_from_config

from .nsde import BaseNSDE
from .nsde import create_nsde_from_config
from .encoders import Encoder
from .bond_net import get_bond_net

from ..configs.config_encoder import EncoderCfg
from ..configs.config_nsde import NSDECfg
from ..configs.config_bondnet import BaseBondNetCfg

from ..types.data_types import EncoderInputs

from ..utils.checks import _check_positive_integer_value
from ..utils.checks import _check_positive_value
from ..utils.artifacts import atomic_save_json
from ..utils.artifacts import _to_jsonable


class ShortRateModel(nn.Module):
    """
    Short Rate Model container.

    Attributes
    ----------
    name : str
        Name of the model.
    nsde : Union[NSDECfg, BaseNSDE]
        Neural SDE model or its configuration.
    encoder : Union[EncoderCfg, Encoder]
        Encoder module or its configuration.
    decoder : Optional[Union[Dict, nn.Module]]
        Decoder module or its configuration dictionary.
        If None, defaults to a linear layer.
    latent_dim : Optional[int]
        Latent dimension of the model. If None, inferred from encoder or nsde.
    input_dim : Optional[int]
        Input dimension for the encoder. If None, inferred from the data.
    noise_dim : int
        Noise dimension passed to the NSDE factory when nsde is a config.

    Initialization idea
    -------------------
    Can be initialized with either:
    - Both encoder and nsde as configs
    - Encoder as a module and nsde as a config
    - Encoder as a config and nsde as a module
    - Both encoder and nsde as modules

    In all cases:
    - If encoder and nsde are BOTH configs, latent_dim must be provided.
    - If only one is a config, latent_dim is inferred from the other.
    - If both are modules, latent_dim must match between them.
    """

    def __init__(
        self,
        name: str,
        nsde: Union[NSDECfg, BaseNSDE],
        encoder: Union[EncoderCfg, Encoder],
        decoder: Optional[Union[Dict, nn.Module]] = None,
        bondnet: Optional[Union[BaseBondNetCfg, nn.Module]] = None,
        latent_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        noise_dim: Optional[int] = None,
        **kwargs
    ):
        
        super().__init__()
        
        self.name = str(name)

        if noise_dim is not None:
            _check_positive_integer_value(noise_dim, "noise_dim")
        if latent_dim is not None:
            _check_positive_integer_value(latent_dim, "latent_dim")
            # Pre-build guard: latent_dim == 1 collapses the decoder and breaks
            # several downstream broadcasts. Reject early to avoid wasted work.
            if int(latent_dim) == 1:
                raise ValueError("latent_dim must be >= 2 (latent_dim=1 is not supported).")

        # ---------------------------------
        # Resolve Encoder / NSDE 
        # ---------------------------------

        encoder_is_cfg = isinstance(encoder, EncoderCfg)
        nsde_is_cfg = isinstance(nsde, NSDECfg)

        encoder_is_module = isinstance(encoder, Encoder)
        nsde_is_module = isinstance(nsde, BaseNSDE)

        if not (encoder_is_cfg or encoder_is_module):
            raise ValueError("encoder must be either an EncoderCfg or an Encoder module.")
        
        if not (nsde_is_cfg or nsde_is_module):
            raise ValueError("nsde must be either an NSDECfg or a BaseNSDE module.")

        encoder_instance : Encoder
        nsde_instance : BaseNSDE

        # Handle all combinations: cfg or module for encoder/nsde
        # Infer latent_dim when possible

        # 1. Both configs -> latent_dim required (and so noise_dim when noise_type='general')
        if encoder_is_cfg and nsde_is_cfg:
            
            if latent_dim is None:
                raise ValueError("latent_dim must be provided when encoder and nsde are both configs.")

            encoder_instance = Encoder(output_dim=latent_dim, input_dim=input_dim, config=encoder)
            nsde_instance = create_nsde_from_config(nsde, latent_dim=latent_dim, noise_dim=noise_dim)


        # 2. Encoder cfg, nsde module -> infer latent_dim from encoder
        elif encoder_is_cfg and not nsde_is_cfg:
            
            try:
                latent_dim_nsde = int(nsde.latent_dim)
            except Exception:
                raise ValueError("Error extracting latent_dim from nsde_model. Make sure it has a 'latent_dim' attribute.")
            
            if latent_dim is not None and latent_dim != latent_dim_nsde:
                raise ValueError(f"latent_dim provided ({latent_dim}) does not match nsde latent_dim ({latent_dim_nsde})")

            encoder_instance = Encoder(output_dim=latent_dim_nsde, input_dim=input_dim, config=encoder)
            nsde_instance = nsde
        

        # 3. Encoder is module, nsde is cfg: infer latent_dim from encoder. noise_dim required when noise_type='general')
        elif not encoder_is_cfg and nsde_is_cfg:
            
            try:
                output_dim_encoder = int(encoder.output_dim)
            except Exception:
                raise ValueError("Error extracting output_dim from encoder. Make sure it has an 'output_dim' attribute.")
            
            if latent_dim is not None and latent_dim != output_dim_encoder:
                raise ValueError(f"latent_dim provided ({latent_dim}) does not match encoder output_dim ({output_dim_encoder})")

            nsde_instance = create_nsde_from_config(nsde, latent_dim=output_dim_encoder, noise_dim=noise_dim)
            encoder_instance = encoder


        # 4. Both are modules: check latent_dim consistency
        else:

            try:
                output_dim_encoder = int(encoder.output_dim)
            except Exception:
                raise ValueError("Error extracting output_dim from encoder. Make sure it has an 'output_dim' attribute.")
            
            try:
                latent_dim_nsde = int(nsde.latent_dim)
            except Exception:
                raise ValueError("Error extracting latent_dim from nsde_model. Make sure it has a 'latent_dim' attribute.")

            if latent_dim is not None and latent_dim != output_dim_encoder:
                raise ValueError(f"latent_dim provided ({latent_dim}) does not match encoder output_dim ({output_dim_encoder})")
            
            if latent_dim is not None and latent_dim != latent_dim_nsde:
                raise ValueError(f"latent_dim provided ({latent_dim}) does not match nsde latent_dim ({latent_dim_nsde})")
            
            if output_dim_encoder != latent_dim_nsde:
                raise ValueError(f"Encoder output_dim ({output_dim_encoder}) != NSDE latent_dim ({latent_dim_nsde})")


            encoder_instance = encoder
            nsde_instance = nsde


        # Classes are now instantiated
        self.encoder = encoder_instance
        self.nsde = nsde_instance

        if encoder_is_cfg:
            try:
                self.preprocess_mode = encoder.preprocess_mode
            except Exception:
                self.preprocess_mode = None
                
        else:
            self.preprocess_mode = kwargs.get("preprocess_mode", None)


        # Post-build guard (covers the case where latent_dim was inferred from
        # an existing encoder/NSDE module).
        if self.latent_dim == 1:
            raise ValueError("latent_dim must be >= 2 (latent_dim=1 is not supported).")

        # ---------------------------------
        # Resolve Decoder
        # ---------------------------------

        if decoder is None:
            self.decoder = nn.Linear(self.latent_dim, 1)

        elif isinstance(decoder, dict):
            self.decoder = create_network_from_config(decoder, input_dim=self.latent_dim, output_dim=1)

        else:
            # If a module, check that input_dim or in_features matches.
            decoder_input_dim = getattr(decoder, 'input_dim', getattr(decoder, 'in_features', None))

            if decoder_input_dim != self.latent_dim:
                raise ValueError(
                    f"Decoder input_dim ({decoder_input_dim}) does not match latent_dim ({self.latent_dim}). "
                    "Provide a decoder whose first layer has 'input_dim' / 'in_features' == latent_dim."
                )

            self.decoder = decoder


        # ---------------------------------
        # Resolve Decoder
        # ---------------------------------

        if isinstance(bondnet, BaseBondNetCfg):
            self.bondnet = get_bond_net(bondnet)
        elif isinstance(bondnet, nn.Module):
            self.bondnet = bondnet
        elif bondnet is None:
            self.bondnet = None
        else:
            raise TypeError(
                f"bondnet must be a BaseBondNetCfg, nn.Module or None; got {type(bondnet).__name__}"
            )


        # ++++++++++++++ Now we have all components +++++++++++++

        # ---------------------------------
        # MetaData / Training Info
        # ---------------------------------

        self.created_at = datetime.now().isoformat()

        # Lightweight, JSON-safe metadata dicts
        self.is_trained: bool = False
        self.training_info: Optional[Dict[str, Any]] = None
        self.finetune_history: list[Dict[str, Any]] = []


    # ------------------------------------------------------------------
    # Factory that accepts dicts (Optuna-friendly)
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(
        cls,
        name: str,
        *,
        encoder: Union[Encoder, EncoderCfg, Mapping[str, Any]],
        nsde: Union[BaseNSDE, NSDECfg, Mapping[str, Any]],
        decoder: Optional[Union[nn.Module, Mapping[str, Any]]] = None,
        bondnet: Optional[Union[BaseBondNetCfg, nn.Module, Mapping[str, Any]]] = None,
        latent_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        noise_dim: Optional[int] = 16,
    ) -> "ShortRateModel":
        """
        Same as __init__, but allows dict overrides for every component.

        Parameters
        ----------
        encoder, nsde, decoder, bondnet
            Either a config (`EncoderCfg`, `NSDECfg`, `BaseBondNetCfg`), a
            module (`Encoder`, `BaseNSDE`, `nn.Module`), or a plain
            ``Mapping[str, Any]`` of overrides — handy for Optuna/grid-search.
        latent_dim, input_dim, noise_dim
            Forwarded to `__init__`.
        """
        if isinstance(encoder, Mapping) and not isinstance(encoder, EncoderCfg):
            try:
                encoder = EncoderCfg(**dict(encoder))
            except Exception as e:
                raise ValueError("Could not build Encoder from dict") from e

        if isinstance(nsde, Mapping) and not isinstance(nsde, NSDECfg):
            try:
                nsde = NSDECfg(**dict(nsde))
            except Exception as e:
                raise ValueError("Could not build NSDE from dict") from e

        if isinstance(decoder, Mapping) and not isinstance(decoder, nn.Module):
            decoder = dict(decoder)  # keep override dict for create_network_from_config

        if isinstance(bondnet, Mapping) and not isinstance(bondnet, BaseBondNetCfg) and not isinstance(bondnet, nn.Module):
            # Distinguish SimpleBondNetCfg vs FiLMBondNetCfg via the keyset.
            keys = set(bondnet.keys())
            from ..configs.config_bondnet import SimpleBondNetCfg, FiLMBondNetCfg
            if {"trunk_n_layers", "film_n_layers", "head_n_layers"} & keys:
                bondnet_cls = FiLMBondNetCfg
            else:
                bondnet_cls = SimpleBondNetCfg
            try:
                bondnet = bondnet_cls(**dict(bondnet))
            except Exception as e:
                raise ValueError(f"Could not build BondNet from dict using {bondnet_cls.__name__}") from e

        return cls(
            name=name,
            encoder=encoder,
            nsde=nsde,
            decoder=decoder,
            bondnet=bondnet,
            latent_dim=latent_dim,
            input_dim=input_dim,
            noise_dim=noise_dim,
        )
    

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, past_data, **kwargs) -> Tensor:
        """
        Encode encoder inputs into the latent state ``z_t``.

        Parameters
        ----------
        past_data : EncoderInputs or Tuple[EncoderInputs, EncoderInputs]
            Input bundle(s). A single `EncoderInputs` for `simple` mode, or a
            (fast, slow) tuple for `hierarchical` mode.
        **kwargs
            Forwarded to `self.encoder(...)`. Supported keys: ``state``,
            ``return_state``, ``return_sequence``.

        Returns
        -------
        Tensor
            Latent state. Shape ``(B, latent_dim)`` by default; if
            ``return_sequence=True``, shape ``(B, T, latent_dim)``. If
            ``return_state=True``, a tuple ``(z, final_state)`` is returned.
        """
        past_data_preprocessed = self._preprocess_encoder_input(past_data)
        return self.encoder(past_data_preprocessed, **kwargs)
    

    def decode(self, embedding: Tensor, r0: Optional[Union[float, Tensor]] = None) -> Tensor:
        """
        Decode latent state(s) into a (path of) short rates.

        Parameters
        ----------
        embedding : Tensor
            Latent state of shape ``(latent_dim,)`` or ``(..., latent_dim)``.
            For full paths this is typically ``(n_paths, n_steps, latent_dim)``.
        r0 : Optional[Union[float, Tensor]]
            If provided, the decoded output is shifted so the value at the
            first time step exactly equals ``r0``. Only meaningful for 3D
            inputs.

        Returns
        -------
        Tensor
            Decoded short rate(s), final dimension = 1.
        """

        # 1D latent vector -> add batch dimension
        out = self.decoder(embedding.unsqueeze(0) if embedding.dim() == 1 else embedding)

        if r0 is None:
            return out

        # The r0 shift assumes a 3D path tensor (n_paths, T, 1).
        # If the input wasn't a full path (e.g. a single latent state), skip
        # the shift rather than indexing into a missing time axis.
        if out.dim() < 3:
            return out

        if isinstance(r0, (float, int)):
            r0_t = torch.tensor(float(r0), device=out.device, dtype=out.dtype)
        else:
            r0_t = r0.to(device=out.device, dtype=out.dtype)

        # Shift all times so that out[:, 0, :] == r0.
        return out + (r0_t - out[:, 0, :]).unsqueeze(1)
    
    
    def simulate(
            self, 
            latent_representation : Tensor, 
            n_paths : int = 500, 
            horizon : float = 5.0,
            ts : Optional[Tensor] = None, 
            dt : Optional[float] = None,
            decode : Optional[bool] = True,
            r0 : Optional[Union[float, Tensor]] = None
            ) -> Tensor:
        """
        Simulate paths forward from a latent representation.

        Parameters
        ----------
        latent_representation : Tensor
            Initial latent representation (latent_dim,) or (1, latent_dim).
        n_paths : int
            Number of Monte Carlo paths.
        horizon : float
            Time horizon in years (used only if ts is None).
        ts : Optional[Tensor]
            Time grid (1D tensor) of timesteps to return. If None, nsde dt is used.
        dt : Optional[float]
            Time step in years. If none, and ts is none, timestep of nsde is used
        decode : bool
            If True, applies decoder to return short-rate paths. Otherwise returns latent paths.

        Returns
        -------
        Tensor
            If decode=True: (n_paths, T)
            If decode=False: (n_paths, T, latent_dim)
        """
        _device = latent_representation.device
        _dtype = latent_representation.dtype

        if ts is None:
            # Default time grid: up to horizon (in years) with daily steps
            if dt is None:
                dt = self.nsde.dt
            else:
                assert isinstance(dt, (int, float)) and dt > 0, "dt must be positive"

            ts = torch.arange(0, horizon + dt, dt, device=_device, dtype=_dtype)
        elif isinstance(ts, Tensor):
            ts = ts.to(device=_device, dtype=_dtype)
        else:
            raise ValueError('ts has to be either None or a torch tensor')

        zs = self.nsde(ts, latent_representation, n_paths)
        
        if not decode:
            return zs
        
        return self.decode(zs, r0=r0).squeeze(-1)      # decode


    @torch.no_grad()
    def sample(self, *args, **kwargs) -> Tensor:
        """
        Alias for simulate() with torch.no_grad().
        """
        return self.simulate(*args, **kwargs)
    

    @staticmethod
    def _maybe_concat_short_rate(curve_history: Tensor, short_rate: Optional[Tensor]) -> Tensor:
        """
        Concatenate short-rate history to curve history if available; otherwise
        return curve history unchanged.

        Handles the case where the short-rate tensor is 1D ``(T,)`` instead of
        the expected ``(T, 1)``.
        """
        if short_rate is None:
            return curve_history
        sr = short_rate.unsqueeze(-1) if short_rate.dim() == 1 else short_rate
        return torch.cat([curve_history, sr], dim=-1)


    def _preprocess_encoder_input(
        self,
        past_data: Union[EncoderInputs, Tuple[EncoderInputs, EncoderInputs]],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Convert one or a pair of `EncoderInputs` into the tensor(s) the
        encoder consumes.

        When `past_data.short_rate` is None, only the curve history is fed to
        the encoder (no concatenation is attempted).
        """
        if self.preprocess_mode is not None and str(self.preprocess_mode).lower() != 'none':
            raise NotImplementedError(f"preprocess_mode={self.preprocess_mode} not supported yet.")

        if self.encoder_type == 'simple':
            # If a dataloader already pre-stacked curve+short_rate into one
            # tensor and exposed it as `curve_history` with short_rate=None,
            # we just forward it.
            return self._maybe_concat_short_rate(past_data.curve_history, past_data.short_rate)

        if self.encoder_type == 'hierarchical':
            fast_data, slow_data = past_data
            out_fast = self._maybe_concat_short_rate(fast_data.curve_history, fast_data.short_rate)
            out_slow = self._maybe_concat_short_rate(slow_data.curve_history, slow_data.short_rate)
            return out_fast, out_slow

        raise ValueError(f"Unknown encoder_type={self.encoder_type}")


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        return int(self.nsde.latent_dim)

    @property
    def noise_dim(self) -> int:
        return int(self.nsde.noise_dim)

    @property
    def encoder_type(self) -> str:
        return str(self.encoder.cfg.mode)
    
    @property
    def nsde_type(self) -> str:
        return str(self.nsde.cfg.type)
    


    # ------------------------------------------------------------------
    # Training + Finetune metadata (SOTA but minimal)
    # ------------------------------------------------------------------

    def cache_training_info(self, training_info: Dict[str, Any]) -> None:
        """
        Stores training metadata and marks model as trained.

        NOTE:
        - Actual optimizer/scheduler states live in checkpoints (ArtifactManager).
        - JSON stores only human-readable training metadata.
        """
        self.is_trained = True
        self.training_info = dict(training_info)


    # def cache_finetune_info(self, finetune_payload: Dict[str, Any]) -> None:
    #     """
    #     Append an entry to finetune_history.

    #     Expected keys (recommended):
    #     - date
    #     - train_window (optional)
    #     - lr
    #     - epochs
    #     - unfrozen (e.g. "decoder_only")
    #     - metric_before / metric_after (optional)
    #     """
    #     payload = dict(finetune_payload)
    #     if "date" not in payload:
    #         payload["date"] = datetime.now().isoformat()
    #     self.finetune_history.append(payload)


    # ------------------------------------------------------------------
    # Model manifest (JSON)
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> Dict[str, Any]:
        """
        JSON-safe model metadata.
        """
        info = {
            "name": self.name,
            "created_at": self.created_at,
            "is_trained": bool(self.is_trained),
            "encoder_type": self.encoder_type,
            "nsde_type": self.nsde_type,
            "latent_dim": self.latent_dim,
            "noise_dim": self.noise_dim,
            "encoder_cfg": _to_jsonable(self.encoder.cfg),
            "torch_version": torch.__version__,
        }

        # Safely store configs even if they contain mappingproxy objects
        try:
            info["nsde_config"] = _to_jsonable(getattr(self.nsde, "cfg", None))
        except Exception:
            pass

        try:
            info["encoder_config"] = _to_jsonable(getattr(self.encoder, "cfg", None))
        except Exception:
            pass

        try:
            info["decoder_config"] = _to_jsonable(getattr(self.decoder, "cfg", None))
        except Exception:
            pass

        if self.training_info is not None:
            info["training_info"] = _to_jsonable(self.training_info)

        # if self.finetune_history is not None:
        #     info["finetune_history"] = _to_jsonable(self.finetune_history)

        return info
        
    
    def save_model_info(self, output_dir: Union[str, os.PathLike]) -> str:
        """
        Write model_info.json (atomic).
        """
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / "model_info.json"
        atomic_save_json(self.model_info, path)
        return str(path)

