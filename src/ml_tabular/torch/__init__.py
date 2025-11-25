"""
PyTorch-related components for ml_tabular.

Subpackages typically include:

- datasets: Dataset implementations for tabular and time-series data.
- models: Neural network architectures for these data types.
- training: Training loops, evaluation helpers, and utilities like early stopping.

In most cases you will import from the subpackages directly:

    from ml_tabular.torch.datasets import tabular
    from ml_tabular.torch.models import tabular_mlp
    from ml_tabular.torch.training import loops
"""

from __future__ import annotations

__all__: list[str] = []
