from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ml_tabular.torch.training.loops import (
    EarlyStopping,
    evaluate,
    fit,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# Helpers: simple synthetic regression problem
# ---------------------------------------------------------------------------


def _make_linear_regression_problem(
    n_samples: int = 64,
    noise_std: float = 0.0,
) -> tuple[TensorDataset, int, int]:
    """Create a tiny y = 3x - 2 regression problem for testing.

    Returns
    -------
    dataset:
        A TensorDataset of (x, y) where x.shape == (n_samples, 1),
        y.shape == (n_samples, 1).
    input_dim:
        Feature dimension (1).
    output_dim:
        Target dimension (1).
    """
    x = torch.linspace(-1.0, 1.0, steps=n_samples).unsqueeze(1)  # (N, 1)
    y = 3.0 * x - 2.0

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    dataset = TensorDataset(x, y)
    return dataset, 1, 1


def _make_model(input_dim: int = 1, output_dim: int = 1) -> nn.Module:
    """Simple one-layer linear regression model."""
    return nn.Sequential(nn.Linear(input_dim, output_dim))


# ---------------------------------------------------------------------------
# train_one_epoch behaviour
# ---------------------------------------------------------------------------


def test_train_one_epoch_reduces_loss() -> None:
    """On a simple regression problem, one epoch of training should reduce loss."""
    dataset, input_dim, output_dim = _make_linear_regression_problem()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = _make_model(input_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    # Compute initial loss manually
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
    initial_loss = total_loss / total_n

    # One training epoch
    avg_train_loss = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        max_batches=None,
        scaler=None,
    )

    assert avg_train_loss < initial_loss


# ---------------------------------------------------------------------------
# evaluate behaviour
# ---------------------------------------------------------------------------


def test_evaluate_matches_manual_average_loss() -> None:
    """evaluate() should compute the same mean loss as a manual loop."""
    dataset, input_dim, output_dim = _make_linear_regression_problem()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = _make_model(input_dim, output_dim)
    loss_fn = nn.MSELoss()

    # Manual evaluation
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)
    manual_avg = total_loss / total_n

    # Using evaluate()
    eval_avg = evaluate(
        model=model,
        dataloader=loader,
        loss_fn=loss_fn,
        device="cpu",
        max_batches=None,
    )

    assert eval_avg == pytest.approx(manual_avg, rel=1e-6)


# ---------------------------------------------------------------------------
# fit behaviour: full training loop + history
# ---------------------------------------------------------------------------


def test_fit_runs_and_returns_history() -> None:
    """fit() should run multiple epochs and return train/val loss history."""
    dataset, input_dim, output_dim = _make_linear_regression_problem()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = _make_model(input_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    num_epochs = 5

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device="cpu",
        early_stopping=None,
        max_train_batches=None,
        max_val_batches=None,
        scheduler=None,
    )

    # History should contain train and val losses
    assert "train_losses" in history
    assert "val_losses" in history

    train_losses: List[float] = history["train_losses"]
    val_losses: List[float] = history["val_losses"]

    assert len(train_losses) == num_epochs
    assert len(val_losses) == num_epochs

    # Expect loss to generally decrease over epochs (not strictly monotonic)
    assert train_losses[-1] < train_losses[0]


# ---------------------------------------------------------------------------
# EarlyStopping behaviour
# ---------------------------------------------------------------------------


def test_early_stopping_min_mode() -> None:
    """EarlyStopping(mode='min') should trigger when metric stops improving."""
    es = EarlyStopping(patience=2, min_delta=0.1, mode="min")

    metrics = [1.0, 0.95, 0.96, 0.97, 0.98]
    stops = []

    for m in metrics:
        stops.append(es.step(m))

    # First few steps should not stop; last one or two likely do.
    # We don't assert exact pattern, just that at some point it requests stop.
    assert any(stops), "Expected EarlyStopping to eventually request stop in 'min' mode."


def test_early_stopping_max_mode() -> None:
    """EarlyStopping(mode='max') should trigger when metric stops improving upward."""
    es = EarlyStopping(patience=2, min_delta=0.05, mode="max")

    metrics = [0.5, 0.55, 0.56, 0.56, 0.55]
    stops = [es.step(m) for m in metrics]

    assert any(stops), "Expected EarlyStopping to eventually request stop in 'max' mode."


def test_early_stopping_resets_on_improvement() -> None:
    """When metric improves beyond min_delta, patience counter should reset."""
    es = EarlyStopping(patience=2, min_delta=0.1, mode="min")

    # Start with a baseline
    assert es.step(1.0) is False  # best = 1.0

    # Slightly worse -> no stop yet
    assert es.step(1.05) is False  # counter = 1
    assert es.step(1.04) is False  # counter = 2 (borderline)

    # Now a clear improvement (1.0 -> 0.85) beyond min_delta; should reset patience
    assert es.step(0.85) is False  # best updated, counter reset

    # A couple more bad epochs before stop
    stop_flags = [es.step(0.9), es.step(0.95), es.step(0.96)]
    assert any(stop_flags)


# ---------------------------------------------------------------------------
# fit + EarlyStopping integration
# ---------------------------------------------------------------------------


def test_fit_respects_early_stopping() -> None:
    """fit() should stop early when EarlyStopping signals it.

    We don't assert the exact epoch; we only require that the history
    length is <= num_epochs and that nothing crashes.
    """
    dataset, input_dim, output_dim = _make_linear_regression_problem()
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = _make_model(input_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    num_epochs = 20
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4, mode="min")

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=num_epochs,
        device="cpu",
        early_stopping=early_stopping,
        max_train_batches=None,
        max_val_batches=None,
        scheduler=None,
    )

    train_losses: List[float] = history["train_losses"]
    val_losses: List[float] = history["val_losses"]

    # Early stopping should never *increase* the number of epochs;
    # at worst it runs all num_epochs.
    assert 1 <= len(train_losses) <= num_epochs
    assert 1 <= len(val_losses) <= num_epochs

    # Loss should still improve compared to the first epoch.
    assert train_losses[-1] <= train_losses[0]
