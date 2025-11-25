"""
Training loops and utilities for ml_tabular.

Typical contents:

- train_one_epoch: single-epoch supervised training loop.
- evaluate: evaluation loop over a DataLoader.
- fit: high-level multi-epoch training with optional scheduler, early stopping,
  and MLflow logging.
- EarlyStopping: simple early-stopping helper.

Import from the module directly when you need behavior control, e.g.:

    from ml_tabular.torch.training.loops import fit, EarlyStopping
"""

from __future__ import annotations

__all__: list[str] = []
