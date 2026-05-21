import traceback

import streamlit as st
import pandas as pd
import numpy as np

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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


def build_umap(df, edit_values=None):
    st.write("### UMAP Dimensionality Reduction")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("No numeric features found for UMAP.")
        return {
            "n_components": 2, "features": [], "n_neighbors": 15, "min_dist": 0.1,
            "scale": "standard", "keep_original": True, "mode": "augment", "keep_cols": []
        }

    default_features = (edit_values.get("features", numeric_columns) if edit_values else numeric_columns)
    default_features = [c for c in default_features if c in numeric_columns]  # safety

    selected_features = st.multiselect(
        "Select numeric features for UMAP",
        numeric_columns,
        default=default_features,
        key="umap_features"
    )

    if not selected_features:
        st.warning("Select at least one feature.")
        return {
            "n_components": 2, "features": [], "n_neighbors": 15, "min_dist": 0.1,
            "scale": "standard", "keep_original": True, "mode": "augment", "keep_cols": []
        }

    # Allow n_components from 1 .. min(#features, 10)
    max_components = min(len(selected_features), 10)
    default_components = (edit_values.get("n_components", min(2, max_components)) if edit_values else min(2, max_components))
    default_components = int(np.clip(default_components, 1, max_components))

    n_components = st.number_input(
        "Number of components",
        min_value=1,
        max_value=int(max_components),
        value=int(default_components),
        step=1,
        key="umap_components"
    )

    n_neighbors_max = max(2, min(200, len(df) - 1))  # neighbors must be < n_samples
    default_neighbors = (edit_values.get("n_neighbors", 15) if edit_values else 15)
    default_neighbors = int(np.clip(default_neighbors, 2, n_neighbors_max))

    n_neighbors = st.number_input(
        "Number of neighbors (must be < number of rows)",
        min_value=2,
        max_value=int(n_neighbors_max),
        value=int(default_neighbors),
        step=1,
        key="umap_neighbors"
    )

    default_min_dist = (edit_values.get("min_dist", 0.1) if edit_values else 0.1)
    min_dist = st.number_input(
        "Minimum distance",
        min_value=0.0,
        max_value=1.0,
        value=float(default_min_dist),
        step=0.01,
        key="umap_min_dist"
    )

    scale = st.selectbox(
        "Scaling",
        ["none", "standard", "robust"],
        index=["none", "standard", "robust"].index(edit_values.get("scale", "standard")) if edit_values else 1,
        key="umap_scaling"
    )

    # Derive keep_original from saved mode key if present (fixes edit-mode state mismatch)
    if edit_values:
        saved_mode = edit_values.get("mode", None)
        if saved_mode is not None:
            _default_keep = (saved_mode == "augment")
        else:
            _default_keep = edit_values.get("keep_original", True)
    else:
        _default_keep = True

    keep_original = st.checkbox(
        "Keep original selected features (don't drop them)",
        value=_default_keep,
        key="umap_keep_original"
    )

    return {
        "n_components": int(n_components),
        "features": selected_features,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "scale": scale,
        "keep_original": bool(keep_original),
        # Derived keys used by apply_umap — kept in sync with keep_original
        "mode": "augment" if keep_original else "strict",
        "keep_cols": []
    }


def apply_umap(df, step):
    df_copy = df.copy()
    try:
        # Guard: dependency check
        if not UMAP_AVAILABLE:
            st.error("umap-learn is not installed. Run: pip install umap-learn")
            return df_copy

        features = step.get("features", [])
        n_components = int(step.get("n_components", 2))
        n_neighbors = int(step.get("n_neighbors", 15))
        min_dist = float(step.get("min_dist", 0.1))
        metric = step.get("metric", "euclidean")
        scale = step.get("scale", "standard")        # none|standard|robust

        # Resolve mode: prefer explicit 'mode' key; fall back to keep_original for backwards compat
        if "mode" in step:
            mode = step["mode"]
        else:
            mode = "augment" if step.get("keep_original", True) else "strict"

        keep_cols = step.get("keep_cols", [])        # keep in strict mode

        if not features:
            st.warning("No features selected for UMAP.")
            return df_copy

        valid = [c for c in features if c in df_copy.columns]
        valid = df_copy[valid].select_dtypes(include=[np.number]).columns.tolist()
        if not valid:
            st.error("UMAP needs numeric features.")
            return df_copy

        n_samples = len(df_copy)
        if n_samples < 3:
            st.error("UMAP needs at least 3 rows.")
            return df_copy

        n_neighbors = min(max(2, n_neighbors), n_samples - 1)
        n_components = min(max(1, n_components), len(valid))

        X = clean_numeric_frame(df_copy[valid])
        X_np = scale_array(X.to_numpy(), scale)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        emb = reducer.fit_transform(X_np)

        cols = [f"UMAP_{i+1}" for i in range(n_components)]
        emb_df = pd.DataFrame(emb, columns=cols, index=df_copy.index)

        if mode == "strict":
            keep_cols = [c for c in keep_cols if c in df_copy.columns]
            out = pd.concat([df_copy[keep_cols], emb_df], axis=1) if keep_cols else emb_df
            out = out.loc[:, ~out.columns.duplicated()]
            st.info(f"✅ UMAP reduced: {len(valid)} → {n_components}")
            return out

        # augment — keep all original columns and add UMAP components
        old = [c for c in df_copy.columns if c.startswith("UMAP_")]
        if old:
            df_copy = df_copy.drop(columns=old)
        out = pd.concat([df_copy, emb_df], axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        st.info(f"✅ UMAP added: +{n_components} components")
        return out

    except Exception as e:
        st.error(f"UMAP failed: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
        return df_copy
