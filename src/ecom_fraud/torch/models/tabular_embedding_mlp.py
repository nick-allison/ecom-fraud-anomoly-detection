from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from ecom_fraud.exceptions import ModelError
from ecom_fraud.logging_config import get_logger

# Reuse the shared activation registry from tabular_mlp
from .tabular_mlp import _get_activation  # type: ignore[attr-defined]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularEmbeddingMLPConfig:
    """Configuration for TabularEmbeddingMLP architecture.

    This configuration explicitly separates numeric and categorical inputs and
    mirrors the "modern MLP" features used in TabularMLP:

      * numeric + categorical embeddings concatenated into a single feature vector
      * hidden_dims / activation / dropout / batch_norm
      * optional residual connections (when shapes allow)
      * optional final_activation for inference

    Parameters
    ----------
    num_numeric_features:
        Number of numeric (dense) features per sample. May be 0 if you want
        a pure-embedding model (not common, but allowed).

    cat_cardinalities:
        Sequence of cardinalities (number of unique values) for each categorical
        feature. Each element corresponds to one column in x_cat.

    cat_embedding_dims:
        Dimension of each embedding. Two options:
          - Explicit: same length as cat_cardinalities.
          - None: we compute a default per-feature dimension using a simple rule
            (see _default_embedding_dims).

    hidden_dims:
        Sizes of hidden layers for the MLP backbone.

    activation:
        Name of activation for hidden layers (uses shared activation registry).

    dropout:
        Dropout probability applied after each activation (except output).

    batch_norm:
        Whether to apply BatchNorm1d after each linear (except output).

    residual:
        If True, we attempt to use residual connections inside the backbone
        where shapes allow. We keep this logic self-contained in this file.

    final_activation:
        Optional activation applied at the very end:
            - None      -> raw logits (recommended for training loops)
            - "sigmoid" -> probabilities for binary / multi-label
            - "softmax" -> probabilities for multi-class (applied along last dim)
            - "tanh"    -> bounded output (-1, 1)
    """

    num_numeric_features: int
    cat_cardinalities: Sequence[int]
    cat_embedding_dims: Optional[Sequence[int]] = None

    hidden_dims: Sequence[int] = (256, 128)
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = True
    residual: bool = False
    final_activation: Optional[Literal["sigmoid", "softmax", "tanh"]] = None

    def validate(self) -> None:
        """Raise ModelError if any config values are invalid."""
        if self.num_numeric_features < 0:
            raise ModelError(
                f"num_numeric_features must be >= 0, got {self.num_numeric_features}",
                code="invalid_model_config",
                context={
                    "field": "num_numeric_features",
                    "value": self.num_numeric_features,
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

        if any(c <= 0 for c in self.cat_cardinalities):
            raise ModelError(
                "All cat_cardinalities must be > 0.",
                code="invalid_model_config",
                context={
                    "field": "cat_cardinalities",
                    "value": list(self.cat_cardinalities),
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

        if self.cat_embedding_dims is not None:
            if len(self.cat_embedding_dims) != len(self.cat_cardinalities):
                raise ModelError(
                    "cat_embedding_dims must have the same length as cat_cardinalities.",
                    code="invalid_model_config",
                    context={
                        "field": "cat_embedding_dims",
                        "value": list(self.cat_embedding_dims),
                        "expected_len": len(self.cat_cardinalities),
                    },
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLPConfig.validate"
                    ),
                )
            if any(d <= 0 for d in self.cat_embedding_dims):
                raise ModelError(
                    "All cat_embedding_dims must be > 0.",
                    code="invalid_model_config",
                    context={
                        "field": "cat_embedding_dims",
                        "value": list(self.cat_embedding_dims),
                    },
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLPConfig.validate"
                    ),
                )

        if any(h <= 0 for h in self.hidden_dims):
            raise ModelError(
                "All hidden_dims must be > 0.",
                code="invalid_model_config",
                context={"field": "hidden_dims", "value": list(self.hidden_dims)},
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

        if not (0.0 <= self.dropout < 1.0):
            raise ModelError(
                "dropout must be in [0.0, 1.0).",
                code="invalid_model_config",
                context={"field": "dropout", "value": self.dropout},
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

        if (
            self.final_activation is not None
            and self.final_activation not in {"sigmoid", "softmax", "tanh"}
        ):
            raise ModelError(
                f"Unsupported final_activation: {self.final_activation!r}",
                code="invalid_model_config",
                context={
                    "field": "final_activation",
                    "value": self.final_activation,
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

        if self.total_input_dim <= 0:
            raise ModelError(
                "Total input dimension (numeric + embeddings) must be > 0.",
                code="invalid_model_config",
                context={
                    "num_numeric_features": self.num_numeric_features,
                    "cat_cardinalities": list(self.cat_cardinalities),
                    "cat_embedding_dims": (
                        list(self.cat_embedding_dims)
                        if self.cat_embedding_dims is not None
                        else None
                    ),
                    "total_input_dim": self.total_input_dim,
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLPConfig.validate"
                ),
            )

    @property
    def resolved_embedding_dims(self) -> Sequence[int]:
        """Return a concrete list of embedding dims for each categorical feature.

        If cat_embedding_dims is provided, we use it. Otherwise we compute
        simple defaults based on the cardinality.
        """
        if self.cat_embedding_dims is not None:
            return self.cat_embedding_dims
        return _default_embedding_dims(self.cat_cardinalities)

    @property
    def total_input_dim(self) -> int:
        """Total feature dimension: numeric + concatenated embeddings."""
        return int(self.num_numeric_features + sum(self.resolved_embedding_dims))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_embedding_dims(cardinalities: Sequence[int]) -> Sequence[int]:
    """Compute default embedding dimensions for each categorical feature.

    You can tweak this heuristic later if you like. For now we use a common
    simple rule-of-thumb: min(50, ceil(cardinality ** 0.5 * 2)) or similar.

    Here we choose a conservative variant:

        dim_i = min(50, max(4, int(round(cardinality ** 0.5 * 2))))

    This ensures:
        - dims grow sub-linearly with cardinality
        - we never go below 4
        - we cap at 50
    """
    dims = []
    for c in cardinalities:
        base = int(round((c ** 0.5) * 2.0))
        dim = max(4, min(50, base))
        dims.append(dim)
    return dims


class ResidualBlock(nn.Module):
    """Simple residual block for MLP layers.

    This is intentionally lightweight and mirrors the spirit of the residual
    behaviour in TabularMLP:

      * If in_dim == out_dim, we use an identity skip: y = x + f(x).
      * If in_dim != out_dim, we use a learned projection on the skip path.

    In both cases, f(x) is a small MLP fragment:
        Linear -> (BatchNorm) -> Activation -> (Dropout)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str,
        dropout: float,
        batch_norm: bool,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.f = nn.Sequential(*layers)

        if in_dim == out_dim:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.proj(x) + self.f(x)


def _build_mlp_backbone(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str,
    dropout: float,
    batch_norm: bool,
    residual: bool,
) -> nn.Module:
    """Build an MLP backbone with optional residual connections.

    Behaviour is intentionally aligned with TabularMLP:

      * If residual is False:
          [Linear -> (BatchNorm) -> Activation -> (Dropout)] x N

      * If residual is True:
          We build a sequence of ResidualBlock modules following hidden_dims.
          This means each block takes in_dim and out_dim from consecutive
          layer sizes; when in_dim != out_dim we use a projection.

    The returned module maps from input_dim to the last hidden dimension
    (or is Identity() if hidden_dims is empty).
    """
    if not hidden_dims:
        return nn.Identity()

    layers: list[nn.Module] = []
    in_dim = input_dim

    if residual:
        # Residual blocks between consecutive feature dimensions
        for h in hidden_dims:
            layers.append(
                ResidualBlock(
                    in_dim=in_dim,
                    out_dim=h,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
            )
            in_dim = h
    else:
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class TabularEmbeddingMLP(nn.Module):
    """Embedding-augmented MLP for tabular data.

    This model is designed to be the "flagship" embedding architecture for
    tabular tasks:

      * Numeric features: x_num of shape (B, num_numeric_features)
      * Categorical features: x_cat of shape (B, n_cat_features) with integer
        indices for each categorical column.
      * Learned embeddings for each categorical feature.
      * Concatenation: [x_num, emb_1, emb_2, ..., emb_k] -> MLP backbone.
      * Optional residual connections in the backbone (when configured).

    Forward API
    -----------
    To stay compatible with the generic training loops, the forward signature
    is designed to accept either:

      1) A single `inputs` argument, where:

         * inputs is a Tensor -> treated as numeric-only (no embeddings).
         * inputs is a tuple/list (x_num, x_cat)
         * inputs is a dict with keys "num" and/or "cat"

      2) Two explicit arguments: forward(x_num, x_cat)

    In practice, when used with EmbeddingTabularDataset, the DataLoader yields
    batches like:

        ((x_num, x_cat), y)

    and after `_move_batch_to_device` the model sees:

        inputs = (x_num_on_device, x_cat_on_device)

    which this forward method handles cleanly.

    Output semantics
    ----------------
    The forward pass returns:

      * raw logits if config.final_activation is None
      * probabilities if final_activation is "sigmoid" or "softmax"
      * bounded continuous values if final_activation is "tanh"

    As with TabularMLP, training loops should generally use logits +
    appropriate loss functions (e.g., BCEWithLogitsLoss, CrossEntropyLoss).
    """

    def __init__(self, config: TabularEmbeddingMLPConfig) -> None:
        super().__init__()

        config.validate()
        self.config = config

        # Build embedding layers
        emb_dims = list(config.resolved_embedding_dims)
        self._embedding_dims = emb_dims  # for introspection

        self.embeddings = nn.ModuleList()
        for idx, (card, dim) in enumerate(
            zip(config.cat_cardinalities, emb_dims, strict=True)
        ):
            # Extra +1 padding / OOV slot can be added later if needed;
            # for now we assume inputs are in [0, card-1].
            emb = nn.Embedding(num_embeddings=card, embedding_dim=dim)
            self.embeddings.append(emb)

        # Build backbone MLP
        backbone_input_dim = config.total_input_dim
        self.backbone = _build_mlp_backbone(
            input_dim=backbone_input_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            residual=config.residual,
        )

        last_hidden_dim = (
            backbone_input_dim if not config.hidden_dims else config.hidden_dims[-1]
        )
        self.output_layer = nn.Linear(last_hidden_dim, 1 if config is None else 1)
        # ^ We'll replace this in a moment with a proper output_dim; using a
        # placeholder here keeps type-checkers quiet.

        # We want a generic interface: output_dim is deduced from last layer
        # dimension; to keep the config small and symmetric with TabularMLP,
        # we compute output_dim from the final Linear layer we construct below.
        # For clarity, we expose an explicit output_dim in the constructor
        # via a dedicated builder (see build_tabular_embedding_mlp).

    @classmethod
    def from_dims(
        cls,
        *,
        num_numeric_features: int,
        cat_cardinalities: Sequence[int],
        cat_embedding_dims: Optional[Sequence[int]] = None,
        output_dim: int = 1,
        hidden_dims: Sequence[int] = (256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
        residual: bool = False,
        final_activation: Optional[Literal["sigmoid", "softmax", "tanh"]] = None,
    ) -> "TabularEmbeddingMLP":
        """Alternative constructor, closer to TabularMLP and the builder pattern."""
        config = TabularEmbeddingMLPConfig(
            num_numeric_features=num_numeric_features,
            cat_cardinalities=tuple(cat_cardinalities),
            cat_embedding_dims=(
                tuple(cat_embedding_dims) if cat_embedding_dims is not None else None
            ),
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
            final_activation=final_activation,
        )
        model = cls(config)
        # Rebuild the output layer now that we know output_dim
        last_hidden_dim = (
            config.total_input_dim if not config.hidden_dims else config.hidden_dims[-1]
        )
        model.output_layer = nn.Linear(last_hidden_dim, output_dim)
        model._final_activation_name = config.final_activation

        logger.info(
            "Initialized TabularEmbeddingMLP",
            extra={
                "num_numeric_features": config.num_numeric_features,
                "cat_cardinalities": list(config.cat_cardinalities),
                "cat_embedding_dims": list(config.resolved_embedding_dims),
                "hidden_dims": list(config.hidden_dims),
                "activation": config.activation,
                "dropout": config.dropout,
                "batch_norm": config.batch_norm,
                "residual": config.residual,
                "final_activation": config.final_activation,
                "total_input_dim": config.total_input_dim,
                "n_parameters": model.count_parameters(trainable_only=True),
            },
        )

        return model

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        """Return the number of parameters in the model."""
        params = (
            p for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_inputs(
        self, inputs: Union[Tensor, Tuple[Tensor, Tensor], dict, None], x_cat: Tensor
    ) -> Tuple[Optional[Tensor], Tensor]:
        """Normalize inputs into (x_num, x_cat) tensors.

        This function centralizes input-shape handling so that the public
        forward signature can stay flexible without becoming messy.

        Cases:
          * inputs is a Tensor and x_cat is provided separately:
              - treat inputs as x_num
          * inputs is a Tensor and x_cat is None:
              - numeric-only model (no embeddings); x_cat must be empty
          * inputs is a tuple/list -> (x_num, x_cat)
          * inputs is a dict -> inputs["num"], inputs["cat"]

        The EmbeddingTabularDataset path is the second bullet: inputs is
        (x_num, x_cat) and x_cat argument is None.
        """
        x_num_tensor: Optional[Tensor] = None
        x_cat_tensor: Optional[Tensor] = None

        if isinstance(inputs, dict):
            x_num_tensor = inputs.get("num", None)
            x_cat_tensor = inputs.get("cat", None)
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ModelError(
                    "TabularEmbeddingMLP expected a tuple/list (x_num, x_cat).",
                    code="invalid_input_structure",
                    context={"len_inputs": len(inputs)},
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP._parse_inputs"
                    ),
                )
            x_num_tensor, x_cat_tensor = inputs
        else:
            # Single tensor path: inputs is x_num; x_cat is from argument
            x_num_tensor = inputs
            x_cat_tensor = x_cat

        # x_cat is required if we have categorical features configured
        if len(self.config.cat_cardinalities) > 0 and x_cat_tensor is None:
            raise ModelError(
                "Categorical features are configured, but x_cat was not provided.",
                code="missing_categorical_inputs",
                context={
                    "num_categorical_features": len(self.config.cat_cardinalities),
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLP._parse_inputs"
                ),
            )

        # Normalize shapes
        if x_num_tensor is not None:
            if x_num_tensor.ndim == 1:
                x_num_tensor = x_num_tensor.unsqueeze(0)
            if x_num_tensor.ndim != 2:
                raise ModelError(
                    "x_num must have shape (batch_size, num_numeric_features).",
                    code="invalid_input_shape",
                    context={"x_num_shape": tuple(x_num_tensor.shape)},
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP._parse_inputs"
                    ),
                )

        if x_cat_tensor is not None:
            if x_cat_tensor.ndim == 1:
                x_cat_tensor = x_cat_tensor.unsqueeze(0)
            if x_cat_tensor.ndim != 2:
                raise ModelError(
                    "x_cat must have shape (batch_size, n_cat_features).",
                    code="invalid_input_shape",
                    context={"x_cat_shape": tuple(x_cat_tensor.shape)},
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP._parse_inputs"
                    ),
                )

        return x_num_tensor, x_cat_tensor

    def _embed_categoricals(self, x_cat: Tensor) -> Tensor:
        """Apply embedding layers to the categorical indices in x_cat.

        x_cat is expected to be (B, n_cat_features) with integer indices.

        We produce a concatenated embedding tensor of shape (B, sum(emb_dims)).
        """
        x_cat = x_cat.long()  # ensure correct dtype for nn.Embedding

        if x_cat.shape[1] != len(self.embeddings):
            raise ModelError(
                "Number of categorical columns does not match number of embeddings.",
                code="invalid_input_shape",
                context={
                    "x_cat_n_features": int(x_cat.shape[1]),
                    "n_embeddings": len(self.embeddings),
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLP._embed_categoricals"
                ),
            )

        batch_size = x_cat.shape[0]
        emb_list: list[Tensor] = []
        for idx, emb_layer in enumerate(self.embeddings):
            col = x_cat[:, idx]  # (B,)
            emb = emb_layer(col)  # (B, emb_dim_i)
            if emb.shape[0] != batch_size:
                raise ModelError(
                    "Embedding output batch size mismatch.",
                    code="embedding_batch_mismatch",
                    context={
                        "col_index": idx,
                        "expected_batch_size": batch_size,
                        "actual_batch_size": int(emb.shape[0]),
                    },
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP._embed_categoricals"
                    ),
                )
            emb_list.append(emb)

        if not emb_list:
            # No categorical features configured -> zero-cat features
            return x_cat.new_zeros((batch_size, 0), dtype=torch.float32)  # type: ignore[arg-type]

        return torch.cat(emb_list, dim=1)

    def _apply_final_activation(self, logits: Tensor) -> Tensor:
        """Apply optional final activation to the output logits."""
        name = getattr(self, "_final_activation_name", None)
        if name is None:
            return logits

        if name == "sigmoid":
            return torch.sigmoid(logits)
        if name == "softmax":
            return torch.softmax(logits, dim=-1)
        if name == "tanh":
            return torch.tanh(logits)

        raise ModelError(
            f"Unsupported final_activation: {name!r}",
            code="invalid_final_activation",
            context={"final_activation": name},
            location=(
                "ecom_fraud.torch.models.tabular_embedding_mlp."
                "TabularEmbeddingMLP._apply_final_activation"
            ),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        inputs: Union[Tensor, Tuple[Tensor, Tensor], dict],
        x_cat: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs:
            One of:
              * Tensor: treated as x_num (numeric features only).
              * Tuple/List: (x_num, x_cat).
              * Dict: {"num": x_num, "cat": x_cat}.

            In the common case with EmbeddingTabularDataset + generic loops,
            this will be a tuple (x_num, x_cat).

        x_cat:
            Optional second argument for categorical indices when `inputs` is
            used for x_num only. This is mainly for ergonomic direct usage and
            is not required when the dataset returns a tuple.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        x_num, x_cat_tensor = self._parse_inputs(inputs, x_cat)

        # Numeric features
        if self.config.num_numeric_features > 0:
            if x_num is None:
                raise ModelError(
                    "Numeric features are configured but x_num was not provided.",
                    code="missing_numeric_inputs",
                    context={
                        "num_numeric_features": self.config.num_numeric_features,
                    },
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP.forward"
                    ),
                )
            if x_num.shape[1] != self.config.num_numeric_features:
                raise ModelError(
                    "x_num feature dimension does not match num_numeric_features.",
                    code="invalid_input_shape",
                    context={
                        "x_num_dim": int(x_num.shape[1]),
                        "num_numeric_features": self.config.num_numeric_features,
                    },
                    location=(
                        "ecom_fraud.torch.models.tabular_embedding_mlp."
                        "TabularEmbeddingMLP.forward"
                    ),
                )
            x_num = x_num.float()
        else:
            x_num = None

        # Categorical embeddings
        if len(self.config.cat_cardinalities) > 0:
            assert x_cat_tensor is not None  # guarded in _parse_inputs
            cat_emb = self._embed_categoricals(x_cat_tensor)
        else:
            # No categorical features configured
            if x_cat_tensor is not None:
                logger.warning(
                    "x_cat was provided but cat_cardinalities is empty; "
                    "categorical inputs will be ignored."
                )
            batch_size = x_num.shape[0] if x_num is not None else 0  # type: ignore[assignment]
            cat_emb = (
                torch.zeros(
                    (batch_size, 0),
                    dtype=torch.float32,
                    device=x_num.device if x_num is not None else None,
                )
                if batch_size > 0
                else torch.zeros((0, 0), dtype=torch.float32)
            )

        # Concatenate numeric + embeddings
        if x_num is not None:
            features = torch.cat([x_num, cat_emb], dim=1)
        else:
            features = cat_emb

        if features.ndim != 2 or features.shape[1] != self.config.total_input_dim:
            raise ModelError(
                "Concatenated feature dimension does not match config.total_input_dim.",
                code="invalid_input_shape",
                context={
                    "features_shape": tuple(features.shape),
                    "expected_total_input_dim": self.config.total_input_dim,
                },
                location=(
                    "ecom_fraud.torch.models.tabular_embedding_mlp."
                    "TabularEmbeddingMLP.forward"
                ),
            )

        backbone_out = self.backbone(features)
        logits = self.output_layer(backbone_out)
        return self._apply_final_activation(logits)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_tabular_embedding_mlp(
    *,
    num_numeric_features: int,
    cat_cardinalities: Sequence[int],
    output_dim: int = 1,
    cat_embedding_dims: Optional[Sequence[int]] = None,
    hidden_dims: Sequence[int] = (256, 128),
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = True,
    residual: bool = False,
    final_activation: Optional[Literal["sigmoid", "softmax", "tanh"]] = None,
) -> TabularEmbeddingMLP:
    """Convenience builder for TabularEmbeddingMLP.

    This mirrors the style of `build_tabular_mlp` and keeps notebooks clean
    and declarative, e.g.:

        model = build_tabular_embedding_mlp(
            num_numeric_features=X_num.shape[1],
            cat_cardinalities=[n_country, n_merchant, n_channel],
            output_dim=1,
            hidden_dims=(512, 256, 128),
            activation="mish",
            dropout=0.1,
            batch_norm=True,
            residual=True,
        )

    Parameters
    ----------
    num_numeric_features:
        Number of columns in the numeric feature matrix X_num.

    cat_cardinalities:
        Cardinalities for each categorical feature (length = n_cat_features).

    output_dim:
        Number of outputs; typically 1 for binary classification, K for
        multi-class, etc.

    cat_embedding_dims:
        Optional explicit embedding dims. If None, defaults are computed.

    hidden_dims, activation, dropout, batch_norm, residual, final_activation:
        See TabularEmbeddingMLPConfig for details.
    """
    model = TabularEmbeddingMLP.from_dims(
        num_numeric_features=num_numeric_features,
        cat_cardinalities=cat_cardinalities,
        cat_embedding_dims=cat_embedding_dims,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        residual=residual,
        final_activation=final_activation,
    )
    return model


__all__ = [
    "TabularEmbeddingMLP",
    "TabularEmbeddingMLPConfig",
    "build_tabular_embedding_mlp",
]
