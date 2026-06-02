"""Models layer: XGBoost inference, CNN + embeddings, thresholds, comparison.

Curated public surface over the historical implementation modules:

* :mod:`channel_heads.models.xgboost`    ← ``inference.xgb``
* :mod:`channel_heads.models.thresholds` ← ``eval.metrics`` (threshold tuning)
* :mod:`channel_heads.models.comparison` — model-variant comparison
* :mod:`channel_heads.models.mars_combined` — Mars Phase-6C combined inference
* :mod:`channel_heads.models.cnn`        ← ``cnn_model`` + ``cnn_training`` (torch)
* :mod:`channel_heads.models.embeddings` ← ``cnn_features`` + ``inference.regime`` (torch)

The CNN/embedding submodules require PyTorch and are imported lazily so this
package imports cleanly without it.
"""

from channel_heads.models import comparison, mars_combined, mars_inference, thresholds, xgboost
from channel_heads.models.comparison import compare_predictions
from channel_heads.models.mars_combined import (
    compare_mars_model_variants,
    run_mars_combined_inference,
    run_model_variant_inference,
    summarize_combined_predictions,
)
from channel_heads.models.mars_inference import run_mars_tabular_inference
from channel_heads.models.thresholds import (
    classification_metrics,
    f1_optimal_threshold,
    max_precision_threshold,
)
from channel_heads.models.xgboost import (
    load_xgb_model,
    predict_with_threshold,
)

# Torch-dependent submodules (optional).
try:
    from channel_heads.models import cnn, embeddings  # noqa: F401
    from channel_heads.models.embeddings import (
        extract_mars_cnn_embeddings,
        load_cnn_model_for_embeddings,
        merge_embeddings_with_tabular_features,
        validate_embedding_table,
    )

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

__all__ = [
    "xgboost",
    "thresholds",
    "comparison",
    "mars_inference",
    "mars_combined",
    "run_mars_tabular_inference",
    "run_mars_combined_inference",
    "run_model_variant_inference",
    "compare_mars_model_variants",
    "summarize_combined_predictions",
    "load_xgb_model",
    "predict_with_threshold",
    "f1_optimal_threshold",
    "max_precision_threshold",
    "classification_metrics",
    "compare_predictions",
]

if _HAS_TORCH:
    __all__ += [
        "extract_mars_cnn_embeddings",
        "load_cnn_model_for_embeddings",
        "merge_embeddings_with_tabular_features",
        "validate_embedding_table",
    ]
