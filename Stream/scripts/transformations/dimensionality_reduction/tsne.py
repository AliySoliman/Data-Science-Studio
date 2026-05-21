import traceback

import streamlit as st
import pandas as pd
import numpy as np

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, RobustScaler

try:
    from transformations.utils.robust_ml import clean_numeric_frame, scale_array
except ImportError:
    # Fallback inline implementations so the file works standalone
    def clean_numeric_frame(df):
        X = df.replace([float('inf'), float('-inf')], float('nan'))
        return X.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0)

    def scale_array(X, mode):
        if mode == "standard":
            return StandardScaler().fit_transform(X)
        if mode == "robust":
            return RobustScaler().fit_transform(X)
        return X


def build_tsne(df, edit_values=None):
    st.write("### t-SNE Dimensionality Reduction")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric features found for t-SNE.")
        return {
            "n_components": 2, "features": [], "perplexity": 30.0, "learning_rate": 200.0,
            "scale": "standard", "mode": "strict", "keep_cols": []
        }

    default_features = (edit_values.get("features", numeric_columns) if edit_values else numeric_columns)
    default_features = [c for c in default_features if c in numeric_columns]

    selected_features = st.multiselect(
        "Select numeric features for t-SNE",
        numeric_columns,
        default=default_features,
        key="tsne_features"
    )

    if not selected_features:
        st.warning("Select at least one feature.")
        return {
            "n_components": 2, "features": [], "perplexity": 30.0, "learning_rate": 200.0,
            "scale": "standard", "mode": "strict", "keep_cols": []
        }

    max_components = min(len(selected_features), 3)  # t-SNE commonly 2-3
    default_components = (edit_values.get("n_components", min(2, max_components)) if edit_values else min(2, max_components))
    default_components = int(np.clip(default_components, 1, max_components))

    n_components = st.number_input(
        "Number of components",
        min_value=1,
        max_value=int(max_components),
        value=int(default_components),
        step=1,
        key="tsne_components"
    )

    # Perplexity must be strictly < n_samples (clamped again at apply-time)
    n_samples = len(df)
    perplexity_max = max(5.0, float(min(50, n_samples - 1))) if n_samples > 1 else 5.0
    default_perp = float(edit_values.get("perplexity", 30.0) if edit_values else 30.0)
    default_perp = float(np.clip(default_perp, 5.0, perplexity_max))

    perplexity = st.number_input(
        "Perplexity (must be < number of rows)",
        min_value=5.0,
        max_value=float(perplexity_max),
        value=float(default_perp),
        step=1.0,
        key="tsne_perplexity"
    )

    learning_rate = st.number_input(
        "Learning rate",
        min_value=10.0,
        max_value=1000.0,
        value=float(edit_values.get("learning_rate", 200.0) if edit_values else 200.0),
        step=10.0,
        key="tsne_learning_rate"
    )

    scale = st.selectbox(
        "Scaling",
        ["none", "standard", "robust"],
        index=["none", "standard", "robust"].index(edit_values.get("scale", "standard")) if edit_values else 1,
        key="tsne_scaling"
    )

    mode = st.selectbox(
        "Output mode",
        ["strict", "augment"],
        index=["strict", "augment"].index(edit_values.get("mode", "strict")) if edit_values else 0,
        help="strict = return only tSNE components (+ keep_cols). augment = keep all columns and add components.",
        key="tsne_mode"
    )

    keep_cols_text = st.text_input(
        "Keep columns (comma-separated, e.g., target column)",
        value=",".join(edit_values.get("keep_cols", [])) if edit_values else "",
        key="tsne_keep_cols"
    )
    keep_cols = [c.strip() for c in keep_cols_text.split(",") if c.strip()]

    return {
        "n_components": int(n_components),
        "features": selected_features,
        "perplexity": float(perplexity),
        "learning_rate": float(learning_rate),
        "scale": scale,
        "mode": mode,
        "keep_cols": keep_cols
    }


def apply_tsne(df, step):
    df_copy = df.copy()
    try:
        # Guard: dependency check
        if not TSNE_AVAILABLE:
            st.error("scikit-learn is not installed or TSNE could not be imported. Run: pip install scikit-learn")
            return df_copy

        features = step.get("features", [])
        n_components = int(step.get("n_components", 2))
        perplexity = float(step.get("perplexity", 30.0))
        learning_rate = float(step.get("learning_rate", 200.0))
        scale = step.get("scale", "standard")      # none|standard|robust
        mode = step.get("mode", "strict")          # strict|augment
        keep_cols = step.get("keep_cols", [])

        if not features:
            st.warning("No features selected for t-SNE.")
            return df_copy

        valid = [c for c in features if c in df_copy.columns]
        valid = df_copy[valid].select_dtypes(include=[np.number]).columns.tolist()
        if not valid:
            st.error("t-SNE needs numeric features.")
            return df_copy

        n_samples = len(df_copy)
        if n_samples < 3:
            st.error("t-SNE needs at least 3 rows.")
            return df_copy

        # Strictly clamp perplexity to be < n_samples (TSNE requires perplexity < n_samples)
        if perplexity >= n_samples:
            perplexity = max(1.0, float(n_samples - 1))
            st.warning(f"Perplexity adjusted to {perplexity} (must be strictly < number of rows).")
        perplexity = max(1.0, perplexity)

        n_components = min(max(1, n_components), 3, len(valid))

        X = clean_numeric_frame(df_copy[valid])
        X_np = scale_array(X.to_numpy(), scale)

        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            init="pca",
            random_state=42
        )
        emb = reducer.fit_transform(X_np)

        cols = [f"tSNE_{i+1}" for i in range(n_components)]
        emb_df = pd.DataFrame(emb, columns=cols, index=df_copy.index)

        if mode == "strict":
            keep_cols = [c for c in keep_cols if c in df_copy.columns]
            out = pd.concat([df_copy[keep_cols], emb_df], axis=1) if keep_cols else emb_df
            out = out.loc[:, ~out.columns.duplicated()]
            st.info(f"✅ t-SNE reduced: {len(valid)} → {n_components}")
            return out

        # augment
        old = [c for c in df_copy.columns if c.startswith("tSNE_")]
        if old:
            df_copy = df_copy.drop(columns=old)
        out = pd.concat([df_copy, emb_df], axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        st.info(f"✅ t-SNE added: +{n_components} components")
        return out

    except Exception as e:
        st.error(f"t-SNE failed: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return df_copy
