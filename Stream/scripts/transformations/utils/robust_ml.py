from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


def make_columns_unique(cols):
    """Return a list of unique column names by suffixing duplicates."""
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__v{seen[c]}")
    return out


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf, fill NaNs robustly; ensures no NaNs remain."""
    X = df.replace([np.inf, -np.inf], np.nan)
    # Median fill per col; if all-NaN -> stays NaN; fill remaining with 0
    X = X.apply(lambda col: col.fillna(col.median()), axis=0).fillna(0)
    return X


def scale_array(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "standard":
        return StandardScaler().fit_transform(X)
    if mode == "robust":
        return RobustScaler().fit_transform(X)
    return X  # "none"


def infer_task_type(y: Optional[pd.Series]) -> Tuple[str, Optional[pd.Series]]:
    """
    Returns: ("unsupervised" | "classification" | "regression", y_clean_or_None)
    Robust heuristic:
    - if y is None -> unsupervised
    - if y is non-numeric -> classification (factorize)
    - if y is numeric but has few unique values -> classification
    - else regression
    """
    if y is None:
        return "unsupervised", None

    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.dropna()
    if y.empty:
        return "unsupervised", None

    if not pd.api.types.is_numeric_dtype(y):
        y_enc = pd.Series(pd.factorize(y)[0], index=y.index)
        return "classification", y_enc

    nunique = y.nunique(dropna=True)
    # Heuristic: small number of unique values => classification
    # (you can tune threshold; 20 works decently)
    if nunique <= 20:
        return "classification", y.astype(int, errors="ignore")

    return "regression", y.astype(float)
