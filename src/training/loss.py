# src/training/loss.py
import torch
from torch import nn
from torch import Tensor

from typing import Literal
from typing import Optional


class YieldCurveLoss(nn.Module):
    """
    Mean squared error for yield curves, with absolute or relative mode.

    Modes
    -----
    - ``absolute`` : residual ``e = pred - target``.
    - ``relative`` : residual ``e = (pred - target) / max(eps, |target|)``
      (per-tenor scaling).

    Weights
    -------
    Optional 1D tensor of shape ``(n_maturities,)`` with non-negative entries.
    Zero entries are masked out; positive entries are normalised by their sum.
    If ``None``, uniform weighting over maturities is used.

    Attributes
    ----------
    mode : str
        Residual mode (``'absolute'`` or ``'relative'``).
    eps : float
        Floor applied to the denominator in relative mode.
    reduction : Optional[str]
        Cross-batch reduction (``'mean'``, ``'sum'``, or ``'none'``).
    weights : Optional[Tensor]
        Optional per-maturity weight tensor (registered as a buffer when given).
    """

    def __init__(
        self,
        mode: Literal['absolute', 'relative'] = 'absolute',
        weights: Optional[Tensor] = None,         
        eps: float = 1e-7,
        reduction: Optional[Literal['mean', 'sum']] = 'mean',
    ):
        super().__init__()

        if mode not in ["absolute", "relative"]:
            raise ValueError("mode must be 'absolute' or 'relative'")
        
        if reduction not in ["mean", "sum", None]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        self.mode = mode
        self.eps = float(eps)
        self.reduction = reduction

        # Register weights as buffer so they move with .to(device) / .cuda()
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32)

            # Validate weights
            if w.dim() != 1:
                raise ValueError("weights must be 1D")
            if (w < 0).any():
                raise ValueError('weights must be nonnegative')
            
            self.register_buffer("weights", w, persistent=True)
        
        else:
            self.weights = None  


    def forward(self, pred: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """
        Compute the yield-curve loss.

        Parameters
        ----------
        pred : Tensor
            Predicted yields, shape ``(n_maturities,)`` or ``(batch, n_maturities)``.
        target : Tensor
            Target yields, same shape as ``pred``.
        weights : Optional[Tensor]
            Optional per-maturity weights overriding ``self.weights``.

        Returns
        -------
        Tensor
            Scalar (``reduction='mean'`` or ``'sum'``) or ``(batch,)`` if
            ``reduction='none'``.
        """
        
        # Ensure batch dimension
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        # Residuals
        if self.mode == "absolute":
            err = pred - target
        else: 
            denom = torch.clamp(target.abs(), min=self.eps)
            err = (pred - target) / denom

        # Weights handling (mask zeros to avoid wasted mults / better normalization)
        w = None
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32, device=pred.device)
        elif self.weights is not None:
            w = self.weights.to(pred.device)

        if w is None:
            # uniform weights over maturities
            # effectively mean over last dim via simple average
            per_curve = (err * err).mean(dim=-1)
        else:
            if w.dim() != 1 or w.size(0) != pred.size(-1):
                raise ValueError("weights must be 1D and match n_maturities")
            mask = w > 0
            if not torch.any(mask):
                # all-zero weights → zero loss by convention
                per_curve = torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
            else:
                w_eff = w[mask]
                e_eff = err[..., mask]
                # normalize by sum of positive weights (per-curve normalization)
                denom_w = w_eff.sum()
                per_curve = (w_eff * (e_eff * e_eff)).sum(dim=-1) / denom_w

        # Reduce across batch
        if self.reduction == "mean":
            out = per_curve.mean()
        elif self.reduction == "sum":
            out = per_curve.sum()
        else:  # "none"
            out = per_curve

        if squeeze_back and self.reduction == "none":
            out = out.squeeze(0)
        return out
