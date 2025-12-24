from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ecom_fraud.exceptions import TrainingError
from ecom_fraud.logging_config import get_logger
from ecom_fraud.mlops.mlflow_utils import (
    log_params as mlflow_log_params,
    log_metrics as mlflow_log_metrics,
    mlflow_is_enabled,
)
from ecom_fraud.torch.models.tabular_embedding_mlp import build_tabular_embedding_mlp
from ecom_fraud.torch.training.loops import EarlyStopping, MetricFn, fit
from ecom_fraud.torch.training.metrics import (
    build_binary_classification_metrics_from_logits,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config & result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomSearchConfig:
    """Configuration for random hyperparameter search.

    Parameters
    ----------
    n_trials:
        Number of random trials to run.
    max_epochs:
        Maximum number of epochs per trial (upper bound; early stopping may
        cut this short).
    patience:
        Patience for EarlyStopping (in epochs) on the primary metric.
    primary_metric:
        *Base* name of the metric to optimize (e.g., "pr_auc", "auc", "loss").
        We will look for ``val_<primary_metric>`` in the epoch summaries
        returned by ``fit`` when selecting the best trial, while early stopping
        monitors the unprefixed metric (e.g. "pr_auc") on validation metrics.
    mode:
        "max" if higher values are better (e.g., AUC, PR-AUC),
        "min" if lower values are better (e.g., loss).
    gradient_accumulation_steps:
        Passed through to ``fit`` / ``train_one_epoch``.
    max_grad_norm:
        Optional gradient clipping norm; passed to ``fit``.
    use_mlflow:
        If True and MLflow is enabled, per-trial hyperparameters and the trial's
        best primary metric will be logged via ``log_params`` / ``log_metrics``.
        Epoch-level metrics are logged by ``fit`` when ``use_mlflow=True``.
        This module does **not** create or manage MLflow runs; callers are
        responsible for wrapping the search in a run context if desired.
    seed:
        Optional random seed to make sampling reproducible.
    """

    n_trials: int
    max_epochs: int
    patience: int = 10
    primary_metric: str = "pr_auc"
    mode: Literal["max", "min"] = "max"
    gradient_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    use_mlflow: bool = True
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.n_trials <= 0:
            raise TrainingError(
                "n_trials must be a positive integer.",
                code="hparam_bad_n_trials",
                context={"n_trials": self.n_trials},
                location=(
                    "ecom_fraud.torch.training.hparam_search."
                    "RandomSearchConfig.validate"
                ),
            )
        if self.max_epochs <= 0:
            raise TrainingError(
                "max_epochs must be a positive integer.",
                code="hparam_bad_max_epochs",
                context={"max_epochs": self.max_epochs},
                location=(
                    "ecom_fraud.torch.training.hparam_search."
                    "RandomSearchConfig.validate"
                ),
            )
        if self.patience <= 0:
            raise TrainingError(
                "patience must be a positive integer.",
                code="hparam_bad_patience",
                context={"patience": self.patience},
                location=(
                    "ecom_fraud.torch.training.hparam_search."
                    "RandomSearchConfig.validate"
                ),
            )
        mode_lower = self.mode.lower()
        if mode_lower not in {"max", "min"}:
            raise TrainingError(
                f"mode must be 'max' or 'min', got {self.mode!r}",
                code="hparam_bad_mode",
                context={"mode": self.mode},
                location=(
                    "ecom_fraud.torch.training.hparam_search."
                    "RandomSearchConfig.validate"
                ),
            )


@dataclass(frozen=True)
class TrialResult:
    """Result for a single hyperparameter trial."""

    trial_index: int
    hparams: Dict[str, Any]
    primary_metric_name: str
    primary_metric_value: float
    fit_result: Dict[str, Any]  # whatever ``fit`` returned (history, best_epoch, ...)

    @property
    def history(self) -> Optional[List[Dict[str, Any]]]:
        """Per-epoch training history for this trial, if present.

        We expect fit_result to look like:

            {
                "last_epoch": {...},
                "history": [ {...}, {...}, ... ],
                "best_epoch": {...}  # optional
            }

        This returns the raw value of fit_result["history"]
        (typically a list-of-dicts), or None if not present.
        """
        hist = self.fit_result.get("history")
        if hist is None:
            return None
        # We don't normalize here; caller can decide how to interpret it.
        return hist

    @property
    def best_epoch_summary(self) -> Optional[Mapping[str, Any]]:
        """Shortcut to the best epoch summary (or last_epoch as fallback)."""
        fr = self.fit_result
        return fr.get("best_epoch") or fr.get("last_epoch")


@dataclass(frozen=True)
class RandomSearchResult:
    """Aggregate result of a random search run."""

    config: RandomSearchConfig
    trials: List[TrialResult]
    best_trial_index: int
    best_hparams: Dict[str, Any]
    best_primary_metric_name: str
    best_primary_metric_value: float
    best_state_dict: Dict[str, Tensor]

    @property
    def best_trial(self) -> TrialResult:
        for t in self.trials:
            if t.trial_index == self.best_trial_index:
                return t
        raise RuntimeError(
            f"Best trial with index {self.best_trial_index} not found in trials list."
        )

    @property
    def best_history(self) -> Optional[List[Dict[str, Any]]]:
        """Per-epoch history for the best trial, if available."""
        return self.best_trial.history

    @property
    def histories(self) -> List[Optional[List[Dict[str, Any]]]]:
        """Per-epoch histories for all trials (parallel to self.trials).

        Mostly useful for debugging or if you ever want to plot
        all trial curves. For the notebook we really just need
        best_history.
        """
        return [t.history for t in self.trials]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_improvement(
    current: float,
    best: Optional[float],
    mode: Literal["max", "min"],
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    return current < best


def _extract_metric_from_fit_result(
    fit_result: Mapping[str, Any],
    *,
    metric_key: str,
    location: str,
) -> tuple[float, Mapping[str, float]]:
    """Extract the best value for a given metric key from a ``fit`` result.

    We expect ``fit_result`` to follow the structure defined in
    :func:`ecom_fraud.torch.training.loops.fit`, e.g.::

        {
            "last_epoch": {...},
            "history": [ {...}, {...}, ... ],
            "best_epoch": {...}  # optional
        }

    Each epoch summary is a dict with keys like ``"train_loss"``, ``"val_loss"``,
    ``"train_pr_auc"``, ``"val_pr_auc"``, etc.

    We look for ``metric_key`` in:

    * ``fit_result["best_epoch"]`` if present,
    * otherwise ``fit_result["last_epoch"]``.

    Returns
    -------
    (best_value, epoch_summary_used)
    """
    epoch_summary = fit_result.get("best_epoch") or fit_result.get("last_epoch")
    if epoch_summary is None:
        raise TrainingError(
            "fit_result must contain 'last_epoch' (and optionally 'best_epoch').",
            code="hparam_fit_result_missing_keys",
            context={"keys": list(fit_result.keys())},
            location=location,
        )

    if metric_key not in epoch_summary:
        raise TrainingError(
            f"Metric key '{metric_key}' not found in epoch summary.",
            code="hparam_primary_metric_missing",
            context={
                "metric_key": metric_key,
                "available_keys": list(epoch_summary.keys()),
            },
            location=location,
        )

    value = float(epoch_summary[metric_key])
    return value, epoch_summary


def _clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, Tensor]:
    """Clone a model's state_dict to CPU for later re-loading."""
    state_dict_cpu: Dict[str, Tensor] = {}
    for k, v in model.state_dict().items():
        state_dict_cpu[k] = v.detach().cpu().clone()
    return state_dict_cpu


def _log_trial_to_mlflow(
    trial_index: int,
    hparams: Dict[str, Any],
    primary_metric_name: str,
    primary_metric_value: float,
) -> None:
    """Log hyperparameters and best metric for a trial to MLflow.

    Assumes that, if MLflow is enabled, the caller has already created an
    active run (e.g., by wrapping the entire search in ``mlflow_run``).
    """
    if not mlflow_is_enabled():
        return

    # Prefix param keys with the trial index so they remain distinguishable.
    params_with_prefix = {
        f"trial_{trial_index}.{k}": v for k, v in hparams.items()
    }
    try:
        mlflow_log_params(params_with_prefix)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to log hyperparameters for trial %d to MLflow: %s",
            trial_index,
            exc,
        )

    # Log a single summary metric for the trial; use trial_index as the step
    # so multiple trials are naturally ordered on the x-axis in MLflow UI.
    metric_key = f"trial_{trial_index}.best_{primary_metric_name}"
    try:
        mlflow_log_metrics({metric_key: primary_metric_value}, step=trial_index)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to log best metric for trial %d to MLflow: %s",
            trial_index,
            exc,
        )


# ---------------------------------------------------------------------------
# Hyperparameter samplers
# ---------------------------------------------------------------------------


def _sample_log_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    """Sample a value from a log-uniform distribution in [low, high]."""
    if low <= 0.0 or high <= 0.0 or not (low < high):
        raise TrainingError(
            "low and high for log-uniform sampling must be > 0 and low < high.",
            code="hparam_log_uniform_bad_range",
            context={"low": low, "high": high},
            location=(
                "ecom_fraud.torch.training.hparam_search._sample_log_uniform"
            ),
        )
    log_low = np.log10(low)
    log_high = np.log10(high)
    return float(10 ** rng.uniform(log_low, log_high))


def sample_tabular_embedding_hparams(
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample a set of hyperparameters for :class:`TabularEmbeddingMLP`.

    Returned keys
    -------------
    - "hidden_dims": tuple[int, ...]
    - "dropout": float
    - "activation": str
    - "batch_norm": bool
    - "residual": bool
    - "lr": float
    - "weight_decay": float
    """

    # 1) Hidden layer patterns: shallow / medium / deeper
    hidden_dims_choices: Sequence[Sequence[int]] = (
        # Lean / shallow heads
        (64,),
        (128,),
        (256,),

        # Medium depth / width
        (128, 64),
        (128, 64, 32),
        (256, 128),
        (256, 128, 64),

        # Wider / deeper (best for residual backbones, but still allowed generally)
        (256, 256, 128),
        (512, 256),
        (512, 256, 128),
        (512, 256, 128, 64),
    )

    # Strong activations for tabular
    activation_choices: Sequence[str] = (
        "relu",
        "leaky_relu",
        "gelu",
        "mish",
    )

    batch_norm_choices: Sequence[bool] = (True, False)
    residual_choices: Sequence[bool] = (False, True)

    # --- Sample architecture shape first ---
    hidden_dims = hidden_dims_choices[int(rng.integers(len(hidden_dims_choices)))]

    # Dropout choices depend on how big the net is
    max_width = max(hidden_dims)
    depth = len(hidden_dims)

    # Smaller / shallower nets: lighter dropout, finer granularity
    dropout_choices_narrow: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3)
    # Wider / deeper nets: ensure some real regularization
    dropout_choices_wide: Sequence[float] = (0.1, 0.2, 0.3, 0.4)

    if (max_width >= 512) or (depth >= 3):
        dropout_choices = dropout_choices_wide
    else:
        dropout_choices = dropout_choices_narrow

    activation = activation_choices[int(rng.integers(len(activation_choices)))]
    dropout = float(dropout_choices[int(rng.integers(len(dropout_choices)))])
    batch_norm = bool(batch_norm_choices[int(rng.integers(len(batch_norm_choices)))])
    residual = bool(residual_choices[int(rng.integers(len(residual_choices)))])

    # Learning rate / weight decay: reasonable ranges for Adam/AdamW-style optimizers
    lr = _sample_log_uniform(rng, low=1e-4, high=3e-3)
    weight_decay = _sample_log_uniform(rng, low=1e-6, high=1e-2)

    hparams: Dict[str, Any] = {
        "hidden_dims": tuple(int(h) for h in hidden_dims),
        "activation": activation,
        "dropout": dropout,
        "batch_norm": batch_norm,
        "residual": residual,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }
    return hparams


def sample_tabular_mlp_hparams(
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample hyperparameters for a plain tabular MLP.

    Currently this mirrors :func:`sample_tabular_embedding_hparams`. It is
    defined separately so that you can diverge the search spaces later without
    touching callers.
    """
    return sample_tabular_embedding_hparams(rng)


def sample_ts_mlp_hparams(
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample hyperparameters for a time-series MLP.

    Currently identical to :func:`sample_tabular_embedding_hparams`, but
    provided as a separate hook for future customization.
    """
    return sample_tabular_embedding_hparams(rng)


# ---------------------------------------------------------------------------
# Generic random search driver
# ---------------------------------------------------------------------------


def run_random_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_builder: Callable[[Dict[str, Any]], nn.Module],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    metrics: Mapping[str, MetricFn],
    device: torch.device,
    sampler: Callable[[np.random.Generator], Dict[str, Any]],
    search_config: RandomSearchConfig,
    *,
    base_hparams: Optional[Dict[str, Any]] = None,
    mlflow_trial_prefix: str = "trial",
) -> RandomSearchResult:
    """Run a generic random search over hyperparameters.

    Parameters
    ----------
    train_loader, val_loader:
        Training and validation :class:`DataLoader` objects. A validation set
        is REQUIRED for hyperparameter search; test data must **not** be used
        here.
    model_builder:
        Callable that takes a hyperparameter dict and returns an
        :class:`nn.Module`.
    loss_fn:
        Loss function used in training (e.g., ``BCEWithLogitsLoss``).
    metrics:
        Mapping of metric name -> function ``(y_true, y_pred_logits)`` used by
        :func:`fit` for both training and validation. For binary classification,
        you can use :func:`build_binary_classification_metrics_from_logits`.
    device:
        Torch device (``cpu`` / ``cuda`` / ``mps``).
    sampler:
        Hyperparameter sampler: ``rng -> hparams``.
    search_config:
        :class:`RandomSearchConfig` describing number of trials, max epochs,
        early stopping behaviour, etc.
    base_hparams:
        Optional dictionary of fixed hyperparameters that should be merged
        into each trial's hparams. Values from the sampler override these
        if keys conflict.
    mlflow_trial_prefix:
        Prefix used when constructing per-trial MLflow metric keys and, if
        desired, MLflow run names in higher-level orchestration.

    Returns
    -------
    RandomSearchResult
        Contains per-trial results plus the best config and state_dict.
    """
    location = "ecom_fraud.torch.training.hparam_search.run_random_search"
    search_config.validate()

    if val_loader is None:
        raise TrainingError(
            "val_loader is required for hyperparameter search.",
            code="hparam_missing_val_loader",
            context={},
            location=location,
        )

    rng = np.random.default_rng(search_config.seed)
    mlflow_enabled = search_config.use_mlflow and mlflow_is_enabled()

    trials: List[TrialResult] = []
    best_metric: Optional[float] = None
    best_trial_index: int = -1
    best_hparams: Dict[str, Any] = {}
    best_state_dict: Dict[str, Tensor] = {}

    # Base name (e.g. "pr_auc") from the config
    primary_metric_base = search_config.primary_metric    
    # Actual key in the epoch summaries / val metrics (e.g. "val_pr_auc")
    primary_metric_name = f"val_{primary_metric_base}"

    logger.info(
        "Starting random search: n_trials=%d, primary_metric=%s (key=%s), mode=%s",
        search_config.n_trials,
        primary_metric_base,
        primary_metric_name,
        search_config.mode,
    )

    for trial_idx in range(1, search_config.n_trials + 1):
        logger.info("=== Trial %d/%d ===", trial_idx, search_config.n_trials)

        # Sample hyperparameters and merge with any fixed base_hparams.
        sampled_hparams = sampler(rng)
        if base_hparams:
            hparams: Dict[str, Any] = {**base_hparams, **sampled_hparams}
        else:
            hparams = dict(sampled_hparams)

        logger.info(
            "Trial %d hyperparameters: %s",
            trial_idx,
            {
                k: (v if not isinstance(v, (list, tuple)) else list(v))
                for k, v in hparams.items()
            },
        )

        # Build model for this trial
        model = model_builder(hparams)
        model.to(device)

        # Optimizer (AdamW by default; can be overridden via model_builder if needed)
        lr = float(hparams.get("lr", 1e-3))
        weight_decay = float(hparams.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Early stopping on validation primary_metric (via val_<metric> key)
        early_stopping = EarlyStopping(
            monitor=primary_metric_base,
            mode=search_config.mode,
            patience=search_config.patience,
        )

        # Run training for this trial. We do *not* manage MLflow runs here;
        # callers can wrap the entire search or individual trials as desired.
        fit_use_mlflow = bool(mlflow_enabled)
        fit_result = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=search_config.max_epochs,
            metrics=metrics,
            scheduler=None,
            scheduler_on="epoch",
            scheduler_metric="val_loss",
            scaler=None,
            max_grad_norm=search_config.max_grad_norm,
            gradient_accumulation_steps=search_config.gradient_accumulation_steps,
            log_interval=None,
            early_stopping=early_stopping,
            use_mlflow=fit_use_mlflow,
        )

        # Extract best metric for this trial (using the *val_* key)
        primary_value, _ = _extract_metric_from_fit_result(
            fit_result,
            metric_key=primary_metric_name,
            location=location,
        )

        logger.info(
            "Trial %d finished with %s=%.6f",
            trial_idx,
            primary_metric_name,
            primary_value,
        )

        trial_result = TrialResult(
            trial_index=trial_idx,
            hparams=hparams,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_value,
            fit_result=dict(fit_result),
        )
        trials.append(trial_result)

        # Track global best
        if _is_improvement(primary_value, best_metric, search_config.mode):
            logger.info(
                "New best trial found: trial=%d, %s=%.6f (prev best=%s)",
                trial_idx,
                primary_metric_name,
                primary_value,
                "None" if best_metric is None else f"{best_metric:.6f}",
            )
            best_metric = primary_value
            best_trial_index = trial_idx
            best_hparams = dict(hparams)
            best_state_dict = _clone_state_dict_to_cpu(model)

        # Optional: per-trial logging to MLflow (within whatever run caller chose)
        if mlflow_enabled:
            _log_trial_to_mlflow(
                trial_index=trial_idx,
                hparams=hparams,
                primary_metric_name=primary_metric_name,
                primary_metric_value=primary_value,
            )

    if best_metric is None or best_trial_index < 0:
        raise TrainingError(
            "Random search did not produce any successful trials.",
            code="hparam_no_successful_trials",
            context={},
            location=location,
        )

    result = RandomSearchResult(
        config=search_config,
        trials=trials,
        best_trial_index=best_trial_index,
        best_hparams=best_hparams,
        best_primary_metric_name=primary_metric_name,
        best_primary_metric_value=float(best_metric),
        best_state_dict=best_state_dict,
    )

    logger.info(
        "Random search complete. Best trial=%d, %s=%.6f",
        result.best_trial_index,
        result.best_primary_metric_name,
        result.best_primary_metric_value,
    )

    return result


# ---------------------------------------------------------------------------
# Project-leaning convenience wrapper: tabular embedding MLP
# ---------------------------------------------------------------------------


def run_tabular_embedding_random_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_numeric_features: int,
    cat_cardinalities: Sequence[int],
    cat_embedding_dims: Sequence[int],
    device: torch.device,
    search_config: RandomSearchConfig,
    *,
    pos_weight: Optional[float] = None,
    sampler: Callable[[np.random.Generator], Dict[str, Any]] = sample_tabular_embedding_hparams,
    base_hparams: Optional[Dict[str, Any]] = None,
    mlflow_trial_prefix: str = "tabular_embedding_trial",
) -> RandomSearchResult:
    """Convenience wrapper for random search on the tabular embedding MLP.

    This is intentionally opinionated for binary classification with logits:

    - Model: :class:`TabularEmbeddingMLP` (``output_dim=1``, ``final_activation=None``).
    - Loss: ``BCEWithLogitsLoss``, with optional class-imbalance ``pos_weight``.
    - Metrics: AUC, PR-AUC, precision, recall, F1 from logits.
    - Primary metric (by default): ``"pr_auc"``.
      Early stopping monitors the base metric on validation metrics, while
      selection of the best trial uses the corresponding ``"val_pr_auc"`` key
      in the epoch summaries.

    Parameters
    ----------
    train_loader, val_loader:
        DataLoaders for training and validation splits.
    num_numeric_features:
        Number of numeric features in ``x_num``.
    cat_cardinalities:
        Cardinality (number of unique values) for each categorical feature.
    cat_embedding_dims:
        Embedding dimension for each categorical feature. Must match
        length of ``cat_cardinalities``.
    device:
        Torch device.
    search_config:
        :class:`RandomSearchConfig` describing search behaviour. Its
        ``primary_metric`` is interpreted as a *base* name (e.g. ``"pr_auc"``);
        the search will monitor that base name for early stopping and use
        ``val_<primary_metric>`` for selecting the best trial.
    pos_weight:
        Optional positive-class weight for :class:`BCEWithLogitsLoss`.
        In imbalanced fraud settings this is typically ``n_neg / n_pos``.
    sampler:
        Hyperparameter sampler. Defaults to :func:`sample_tabular_embedding_hparams`.
    base_hparams:
        Optional fixed hyperparameters to merge into each sampled config.
    mlflow_trial_prefix:
        String prefix passed down to :func:`run_random_search` for metric
        naming in MLflow.

    Returns
    -------
    RandomSearchResult
        Best hyperparameters, state_dict, and per-trial results.
    """
    location = (
        "ecom_fraud.torch.training.hparam_search."
        "run_tabular_embedding_random_search"
    )

    if len(cat_cardinalities) != len(cat_embedding_dims):
        raise TrainingError(
            "cat_cardinalities and cat_embedding_dims must have the same length.",
            code="hparam_cat_dims_mismatch",
            context={
                "len_cat_cardinalities": len(cat_cardinalities),
                "len_cat_embedding_dims": len(cat_embedding_dims),
            },
            location=location,
        )

    # --------------------------------------------------------------
    # Loss: BCEWithLogits with optional pos_weight
    # We wrap it so (N,1) logits and (N,) targets both work cleanly.
    # --------------------------------------------------------------
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )
        base_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        base_bce = nn.BCEWithLogitsLoss()

    def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (N, 1) or (N,)
        # targets: (N,) or (N, 1) from the Dataset
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        return base_bce(logits_flat, targets_flat)

    # Metric set: standard binary classification metrics from logits
    metrics = build_binary_classification_metrics_from_logits()

    # Model builder closure: wires dataset-specific dimensions + sampled hparams
    def model_builder(hparams: Dict[str, Any]) -> nn.Module:
        return build_tabular_embedding_mlp(
            num_numeric_features=num_numeric_features,
            cat_cardinalities=list(cat_cardinalities),
            cat_embedding_dims=list(cat_embedding_dims),
            output_dim=1,
            hidden_dims=hparams["hidden_dims"],
            activation=hparams["activation"],
            dropout=hparams["dropout"],
            batch_norm=hparams["batch_norm"],
            residual=hparams.get("residual", False),
            final_activation=None,
        )

    result = run_random_search(
        train_loader=train_loader,
        val_loader=val_loader,
        model_builder=model_builder,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        sampler=sampler,
        search_config=search_config,
        base_hparams=base_hparams,
        mlflow_trial_prefix=mlflow_trial_prefix,
    )
    return result


__all__ = [
    "RandomSearchConfig",
    "TrialResult",
    "RandomSearchResult",
    "sample_tabular_embedding_hparams",
    "sample_tabular_mlp_hparams",
    "sample_ts_mlp_hparams",
    "run_random_search",
    "run_tabular_embedding_random_search",
]
