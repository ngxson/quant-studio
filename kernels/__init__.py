"""quant-studio kernel registry: one module per quant family, merged here."""

from .common import QuantSpec, TORCH_DTYPES
from . import iq2, q4_0

QUANT_TYPES = {**q4_0.SPECS, **iq2.SPECS}
