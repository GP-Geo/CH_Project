"""High-level, readable pipeline functions — the project's public top layer.

These functions make the project explainable top-to-bottom without opening
``scripts/``:

Mars cross-planet inference (:mod:`channel_heads.pipelines.mars`)::

    build_mars_topology -> extract_mars_pairs -> build_mars_features
    -> run_mars_xgb_inference -> build_mars_cnn_patches
    -> extract_mars_cnn_embeddings -> run_mars_combined_inference
    -> compare_mars_model_outputs

Earth training (:mod:`channel_heads.pipelines.earth`)::

    train_earth_cnn -> train_earth_xgb_variants

Figures (:mod:`channel_heads.pipelines.poster`)::

    generate_poster_figures

Mars inference stages are package-resident. Earth training still delegates to
historical scripts pending a separate refactor (see ``docs/architecture.md``).
"""

from channel_heads.pipelines import earth, mars, poster
from channel_heads.pipelines.earth import (
    train_earth_cnn,
    train_earth_models,
    train_earth_xgb_variants,
)
from channel_heads.pipelines.mars import (
    build_mars_cnn_patches,
    build_mars_features,
    build_mars_topology,
    compare_mars_model_outputs,
    extract_mars_cnn_embeddings,
    extract_mars_pairs,
    run_full_mars_pipeline,
    run_mars_combined_inference,
    run_mars_xgb_inference,
)
from channel_heads.pipelines.poster import generate_poster_figures

__all__ = [
    "mars",
    "earth",
    "poster",
    # Mars stages
    "build_mars_topology",
    "extract_mars_pairs",
    "build_mars_features",
    "run_mars_xgb_inference",
    "build_mars_cnn_patches",
    "extract_mars_cnn_embeddings",
    "run_mars_combined_inference",
    "compare_mars_model_outputs",
    "run_full_mars_pipeline",
    # Earth training
    "train_earth_cnn",
    "train_earth_xgb_variants",
    "train_earth_models",
    # Figures
    "generate_poster_figures",
]
