# src/utils/checks.py
"""
Tiny validation helpers used throughout the configs and constructors.

Each helper raises ``ValueError`` with a message that names the offending
parameter, so call sites can stay short and uniform.
"""
from numbers import Number
from typing import Any

import numpy as np
from torch import Tensor


def _check_positive_value(scalar: Number, name: str) -> None:
    """
    Ensure that ``scalar > 0``.

    Parameters
    ----------
    scalar : Number
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if scalar <= 0:
        raise ValueError(f"The parameter '{name}' should be positive.")


def _check_positive_integer_value(scalar: Number, name: str) -> None:
    """
    Ensure that ``scalar`` is an int (or numpy integer) strictly greater than 0.

    Parameters
    ----------
    scalar : Number
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if not isinstance(scalar, (int, np.integer)) or scalar <= 0:
        raise ValueError(f"The parameter '{name}' should be a positive integer.")


def _check_non_negative_value(scalar: Number, name: str) -> None:
    """
    Ensure that ``scalar >= 0``.

    Parameters
    ----------
    scalar : Number
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if scalar < 0:
        raise ValueError(f"The parameter '{name}' should be non negative.")


def _check_boolean(boolean: bool, name: str) -> None:
    """
    Ensure that ``boolean`` is a Python ``bool``.

    Parameters
    ----------
    boolean : bool
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if not isinstance(boolean, bool):
        raise ValueError(f"The parameter '{name}' should be a boolean.")


def _check_string(string: str, name: str) -> None:
    """
    Ensure that ``string`` is a Python ``str``.

    Parameters
    ----------
    string : str
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if not isinstance(string, str):
        raise ValueError(f"The parameter '{name}' should be a string.")


def _check_is_tensor_or_none(x: Any, name: str) -> None:
    """
    Ensure that ``x`` is either a ``torch.Tensor`` or None.

    Parameters
    ----------
    x : Any
        Value to validate.
    name : str
        Name reported in the error message.
    """
    if x is not None and not isinstance(x, Tensor):
        raise ValueError(f"The parameter '{name}' should be a torch Tensor or None.")
