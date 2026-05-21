import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, cast

# ___________________________________________________________________________________________________________________________________________________________

def infer_task_type(df: pd.DataFrame, target_column: Optional[str]) -> str:
    """
    Determine ML task type purely in Python — the LLM must never guess this.
    Rules:
      - No target → 'Unsupervised'
      - Target is object/bool/category OR has <=20 unique integer values → 'Classification'
      - Target is float OR has >20 unique integer values → 'Regression'
    """
    if not target_column or target_column not in df.columns:
        return "Unsupervised"
    col = df[target_column]
    if col.dtype == "object" or col.dtype.name in ("category", "bool"):
        return "Classification"
    if pd.api.types.is_integer_dtype(col):
        return "Classification" if col.nunique() <= 20 else "Regression"
    return "Regression"  # float

# ___________________________________________________________________________________________________________________________________________________________

def infer_data_risks(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute essential datasets risks and statistical characteristics.
    
    This function analyzes the input DataFrame to detect issues such as 
    missing values, duplicates, high correlations, outliers, cardinality, 
    and class imbalance. It groups these findings into a unified profile 
    dictionary that serves as the factual grounding for downstream models.

    Args:
        df (pd.DataFrame): The main dataset to analyze.
        target_column (Optional[str]): The column name designated as the target 
                                       variable, if any.

    Returns:
        Dict[str, Any]: A nested dictionary detailing the dataset's risks 
                        and compositional facts.
    """
    profile: Dict[str, Any] = {}
    profile["duplicates"] = int(df.duplicated().sum())
    profile["missing_total"] = int(df.isnull().sum().sum())
    profile["missing_pct"] = round((df.isnull().sum().sum() / max(df.size, 1)) * 100, 2)
    # Fully-missing columns should be tracked separately from constant columns
    profile["fully_missing_cols"] = [c for c in df.columns if int(df[c].isnull().sum()) == len(df)]
    # Constant columns exclude fully-missing columns
    profile["constant_cols"] = [
        c for c in df.columns
        if c not in profile["fully_missing_cols"] and df[c].nunique(dropna=False) <= 1
    ]
    
    # ── Dataset scale (#12) ────────────────────────────────────────────────────
    n_rows = len(df)
    if n_rows < 1000:
        profile["dataset_scale"] = "small"
        profile["size_summary"] = f"small dataset ({n_rows} rows)"
    elif n_rows < 10000:
        profile["dataset_scale"] = "medium"
        profile["size_summary"] = f"medium dataset ({n_rows} rows)"
    else:
        profile["dataset_scale"] = "large"
        profile["size_summary"] = f"large dataset ({n_rows} rows)"

    # ── Identifier-like columns (Rule 4: Stronger Detection) ──────────────────
    id_like_cols = []
    id_tokens = {"id", "uuid", "guid", "pk", "no", "nr", "index", "sno", "rownum", "invoice", "record"}
    
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    for col in df.columns:
        if col in profile["fully_missing_cols"] or col in profile["constant_cols"]:
            continue
            
        name_lower = col.lower()
        clean_name = re.sub(r'[^a-z]', '', name_lower)
        nunique = df[col].nunique(dropna=True)
        unique_ratio = nunique / n_rows if n_rows > 0 else 0
        
        is_id = False
        # 1. Monotonic increasing integers (indices)
        if pd.api.types.is_integer_dtype(df[col]) and nunique == n_rows:
            vals = df[col].sort_values()
            if len(vals) > 1:
                diffs = vals.diff().dropna()
                if bool(np.all(diffs.values == 1)):
                    is_id = True
        
        # 2. UUID-like strings
        if not is_id and df[col].dtype == "object" and unique_ratio > 0.9:
            sample = df[col].dropna().head(10).astype(str)
            if any(re.match(uuid_pattern, val.lower()) for val in sample):
                is_id = True
                
        # 3. Name token + High uniqueness
        if not is_id and any(tok in clean_name for tok in id_tokens) and unique_ratio > 0.95:
            is_id = True
            
        if is_id:
            id_like_cols.append(col)
            
    profile["id_like_cols"] = sorted(set(id_like_cols))

    # ── Specialized Column Types (Rule 2, 3, 5: Semantic mapping) ─────────────
    profile["name_like_cols"] = []
    profile["date_like_cols"] = []
    profile["text_like_cols"] = []
    profile["potential_leakage_cols"] = []
    
    name_tokens = {"name", "firstname", "lastname", "fullname", "surname", "customer", "staff", "person"}
    date_tokens = {"date", "time", "timestamp", "created", "updated", "start", "end", "dob", "birth"}
    text_tokens = {"comment", "desc", "review", "note", "address", "summary", "feedback"}

    for col in df.columns:
        # Exclusion logic (Rule 10): skip if already dropped/categorized
        if (col in profile["id_like_cols"] or 
            col in profile["constant_cols"] or 
            col in profile["fully_missing_cols"] or 
            col == target_column):
            continue
            
        name_lower = col.lower()
        clean_name = re.sub(r'[^a-z]', '', name_lower)
        nunique = df[col].nunique(dropna=True)
        unique_ratio = nunique / n_rows if n_rows > 0 else 0
        is_obj = (df[col].dtype == "object")
        
        # 2. Names (Rule 3: Token + High uniqueness + Object)
        if is_obj and any(t in clean_name for t in name_tokens):
            if unique_ratio > 0.5: # Names are usually quite unique but not 100% (common names)
                profile["name_like_cols"].append(col)
                continue

        # 3. Dates (Rule 2: Hint AND Parseable)
        if any(t in clean_name for t in date_tokens):
            # Strict check: if numeric, it's likely a duration/count unless it's a huge timestamp
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].max() < 10**10:
                continue

            # Try to parse a sample
            sample = df[col].dropna().head(50)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    # Require >70% success rate on sample
                    if parsed.notnull().mean() > 0.7:
                        # Distinguish date-only vs timestamp
                        has_time = False
                        # Check if any have non-zero hour/min
                        # Use .dt accessor or time attributes
                        if any(p.hour != 0 or p.minute != 0 for p in parsed if pd.notnull(p)):
                            has_time = True
                        
                        profile["date_like_cols"].append({
                            "column": col,
                            "type": "timestamp" if has_time else "date-only"
                        })
                        continue
                except:
                    pass

        # 4. Text (Rule 5: Average length, unique count, tokens, or whitespace)
        if is_obj:
            sample_str = df[col].dropna().head(50).astype(str)
            if not sample_str.empty:
                avg_len = sample_str.str.len().mean()
                has_spaces = sample_str.str.contains(" ").mean() > 0.5
                if (avg_len > 40 and unique_ratio > 0.8) or any(t in clean_name for t in text_tokens) or (avg_len > 100):
                    profile["text_like_cols"].append(col)
                    continue
    

    # ── Final Column Role Filtering (The "One Exclusion Set" Rule) ─────────────
    date_col_names = [d["column"] if isinstance(d, dict) else d for d in profile["date_like_cols"]]
    drop_parse_set = (set(profile["fully_missing_cols"]) | set(profile["constant_cols"]) | 
                      set(profile["id_like_cols"]) | set(profile["name_like_cols"]) | 
                      set(profile["text_like_cols"]) | set(date_col_names))
    if target_column:
        drop_parse_set.add(target_column)

    # ── Feature-only view ─────────────────────────────────────────────────────
    feature_df = df.drop(columns=list(drop_parse_set), errors="ignore")
    
    # Initialize categorization lists and populate them
    profile["categorical_cols"] = []
    profile["binary_categorical_cols"] = []
    profile["numeric_cols"] = []
    
    for col in feature_df.columns:
        nu = col_nunique = feature_df[col].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            if nu <= 10:
                 profile["categorical_cols"].append(col)
                 if nu == 2: profile["binary_categorical_cols"].append(col)
            else:
                 profile["numeric_cols"].append(col)
        else:
            profile["categorical_cols"].append(col)
            if nu == 2: profile["binary_categorical_cols"].append(col)

    # Moved categorical population and feature_df creation up to provide better context for analysis

    # ── Redundancy & Rescaling (Rule 7) ────────────────────────────────────────
    redundant_notes = []
    num_only = feature_df.select_dtypes(include=["number"])
    if len(num_only.columns) >= 2:
        for i, c1 in enumerate(num_only.columns):
            for c2 in num_only.columns[i+1:]:
                # Check for linear rescaling: c1 = a*c2 + b
                # We can use correlation for this, but also check if ratio is constant
                s1 = num_only[c1]
                s2 = num_only[c2]
                c_val = float(s1.corr(s2))
                if abs(c_val) > 0.999:
                    redundant_notes.append(f"Rescaling/Duplication: `{c1}` and `{c2}` are nearly identical (rho={c_val:.4f}). Consider dropping one.")
    profile["redundant_feature_notes"] = redundant_notes
    
    # Existing correlation clusters logic remains useful
    high_corr_pairs = []
    if not num_only.empty:
        corr_matrix = num_only.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > 0.85:
                    high_corr_pairs.append((row, col))
                    
    # Build correlation clusters (connected components)
    adj = {}
    for u, v in high_corr_pairs:
        if u not in adj:
            adj[u] = set()
        if v not in adj:
            adj[v] = set()
        adj[u].add(v)
        adj[v].add(u)
        
    clusters = []
    visited = set()
    for node in adj:
        if node not in visited:
            comp = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in comp:
                    comp.add(curr)
                    if curr in adj:
                        stack.extend(adj[curr] - comp)
            visited.update(comp)
            if len(comp) > 1:
                clusters.append(sorted([str(x) for x in comp]))
                
    profile["correlation_clusters"] = clusters
    profile["high_correlations"] = [", ".join(c) for c in clusters][:10]

    # ── Feature counts & Dimensionality ───────────────────────────────────────
    n_features = len(feature_df.columns)
    profile["numeric_feature_count"] = len(feature_df.select_dtypes(include=["number"]).columns)
    profile["categorical_feature_count"] = len(feature_df.select_dtypes(exclude=["number"]).columns)

    if n_features <= 10:
        profile["dimensionality"] = "low feature dimensionality"
    elif n_features <= 50:
        profile["dimensionality"] = "moderate feature dimensionality"
    else:
        profile["dimensionality"] = "high feature dimensionality"

    # ── Cardinality buckets (#14) ─────────────────────────────────────────────
    high_card, med_card, low_card = [], [], []
    for col in feature_df.select_dtypes(include=["object", "category", "bool"]):
        nunique = feature_df[col].nunique(dropna=True)
        if nunique <= 5:
            low_card.append(col)
        elif nunique <= 20:
            med_card.append(col)
        else:
            high_card.append(col)
    profile["low_cardinality_cats"]    = low_card
    profile["medium_cardinality_cats"] = med_card
    profile["high_cardinality_cats"]   = high_card

    # ── Rare Category Summary ────────────────────────────────────────────────
    rare_cats = {}
    for col in feature_df.select_dtypes(include=["object", "category"]):
        vc = feature_df[col].value_counts(normalize=True)
        minority_cats = vc[vc < 0.01] # categories with < 1% frequency
        if not minority_cats.empty:
            rare_cats[col] = {
                "count": len(minority_cats),
                "total_pct": float(round(minority_cats.sum() * 100, 1))
            }
    profile["rare_category_summary"] = rare_cats
    

    # ── Skewness & outliers ────────────────────────────────────────────────────
    skewed, outliers = [], []
    num_cols = feature_df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        try:
            skewness = feature_df[num_cols].skew()
            skewed = skewness[skewness.abs() > 1].index.tolist()
        except Exception:
            pass
        for col in num_cols:
            s = feature_df[col].dropna()
            if len(s) < 8:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            if ((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).mean() > 0.05:
                outliers.append(col)
    profile["skewed_numeric_cols"]  = skewed
    profile["outlier_numeric_cols"] = outliers

    # ── Feature families (#11) ────────────────────────────────────────────────
    # Detect columns that share a common prefix stem (e.g., radius_mean/radius_se/radius_worst)
    family_map: Dict[str, list] = {}
    for col in feature_df.columns:
        # strip trailing _suffix patterns
        stem = re.sub(r"[_\s]+(mean|se|std|worst|min|max|sum|count|pct|avg|var)$", "", col.lower())
        if stem != col.lower():
            family_map.setdefault(stem, []).append(col)
    # keep only families with ≥2 members
    profile["feature_families"] = {
        stem: members for stem, members in family_map.items() if len(members) >= 2
    }

    # ── Target distribution & imbalance ───────────────────────────────────────
    profile["target_distribution"] = {}
    profile["largest_class_pct"] = None
    profile["majority_class_label"] = None
    profile["majority_class_pct"] = None
    profile["minority_class_label"] = None
    profile["minority_class_pct"] = None
    profile["imbalance_flag"] = False
    profile["imbalance_level"] = "none"
    profile["imbalance_summary"] = None

    if target_column and target_column in df.columns:
        vc = df[target_column].value_counts(normalize=True, dropna=False)

        distribution = {
            str(k): float(round(float(v) * 100.0, 1))
            for k, v in vc.items()
        }
    
        profile["target_distribution"] = distribution

        if len(vc) > 0:
            majority_label = str(vc.idxmax())
            minority_label = str(vc.idxmin())
            majority_pct = float(round(float(vc.max()) * 100.0, 1))
            minority_pct = float(round(float(vc.min()) * 100.0, 1))

            profile["largest_class_pct"] = majority_pct
            profile["majority_class_label"] = majority_label
            profile["majority_class_pct"] = majority_pct
            profile["minority_class_label"] = minority_label
            profile["minority_class_pct"] = minority_pct

        # Initialize local variables to prevent UnboundLocalError
        imbalance_level = "none"
        imbalance_flag = False

        task = infer_task_type(df, target_column)
        if task == "Classification" and len(vc) >= 2:
            maj = float(vc.max())

            if maj >= 0.80:
                imbalance_level = "severe"
                imbalance_flag = True
            elif maj >= 0.60:
                imbalance_level = "moderate"
                imbalance_flag = True
            else:
                imbalance_level = "none"
                imbalance_flag = False

        profile["imbalance_flag"] = imbalance_flag
        profile["imbalance_level"] = imbalance_level
        profile["imbalance_summary"] = {
            "target_distribution": distribution,
            "imbalance_detected": imbalance_flag,
            "imbalance_level": imbalance_level,
            "majority_class_label": profile["majority_class_label"],
            "majority_class_pct": profile["majority_class_pct"],
            "minority_class_label": profile["minority_class_label"],
            "minority_class_pct": profile["minority_class_pct"],
        }
    
    # ── Potential Leakage (Strict Stage) ──────────────────────────────────────
    leakage_cols = []
    if target_column and target_column in df.columns:
        target_series = df[target_column]
        if pd.api.types.is_numeric_dtype(target_series):
            for col in feature_df.select_dtypes(include=["number"]).columns:
                corr = abs(target_series.corr(df[col]))
                if corr > 0.99: # Sticking to strict 0.99 for actual leakage
                    leakage_cols.append(col)
    profile["potential_leakage_cols"] = sorted(leakage_cols)
    
    # Append cluster info to redundant notes
    for cluster in clusters:
        profile["redundant_feature_notes"].append(f"Cluster: {', '.join(cluster)} (highly redundant group)")
        
    return profile


# ___________________________________________________________________________________________________________________________________________________________

def generate_pipeline_decisions(df: pd.DataFrame, risks: Dict[str, Any], target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply a strict 5-step priority pipeline to every column in the dataset.
    This ensures deterministic preprocessing decisions in Python, leaving 
    the LLM to only explain the rationale.

    Steps:
    1. Drop (Missing, Constant, Identifier, Name, Leakage)
    2. Parse (Date, Text)
    3. Categorical Encode (Binary, Nominal, High-Cardinality)
    4. Numeric Transform (Skewed, Outlier-heavy, Scaling)
    5. Engineering (Derived family/temporal features)
    """
    decisions: Dict[str, Any] = {
        "drop": {},      # col -> reason
        "parse": {},     # col -> format
        "encode": {},    # col -> strategy
        "transform": {}, # col -> strategy
        "engineer": []   # list of suggested derivation rules
    }

    # Reference sets from risks
    miss_set    = set(risks.get("fully_missing_cols", []))
    const_set   = set(risks.get("constant_cols", []))
    id_set      = set(risks.get("id_like_cols", []))
    name_set    = set(risks.get("name_like_cols", []))
    leak_set    = set(risks.get("potential_leakage_cols", []))
    text_set    = set(risks.get("text_like_cols", []))
    
    # Date info is now a list of dicts
    date_info_list = risks.get("date_like_cols", [])
    date_map = {d["column"]: d["type"] for d in date_info_list if isinstance(d, dict)}
    date_set = set(date_map.keys())

    skew_set    = set(risks.get("skewed_numeric_cols", []))
    outlier_set = set(risks.get("outlier_numeric_cols", []))
    redundant_set = set() # We can extract from notes if needed, but usually redundant are dropped
    
    # 1. Pipeline Pass
    for col in df.columns:
        if col == target_column:
            continue
            
        # STEP 1: DROP (Rule 10: Centralized Exclusion)
        if col in miss_set:
            decisions["drop"][col] = "100% missing values"
            continue
        if col in const_set:
            decisions["drop"][col] = "constant (zero variance)"
            continue
        if col in id_set:
            decisions["drop"][col] = "identifier/index field (non-predictive)"
            continue
        if col in name_set:
            decisions["drop"][col] = "personal name / PII (non-predictive)"
            continue
        if col in leak_set:
            decisions["drop"][col] = "potential leakage (perfect correlation with target)"
            continue

        # STEP 2: PARSE (Rule 11: Dedicated Parsing)
        if col in date_set:
            d_type = date_map[col]
            if d_type == "timestamp":
                decisions["parse"][col] = "Extract Year, Month, Day, Weekday, and Hour"
            else:
                decisions["parse"][col] = "Extract Year, Month, Day, Weekday, and Tenure"
            continue
        if col in text_set:
            decisions["parse"][col] = "NLP processing (TF-IDF or Embeddings)"
            continue

        # STEP 3: ENCODE (Rule 6: Semantic-first)
        if col in risks["categorical_cols"]:
            nu = df[col].nunique(dropna=True)
            has_rare = col in risks.get("rare_category_summary", {})
            rare_note = " (with rare levels grouped)" if has_rare else ""
            
            if col in risks["binary_categorical_cols"]:
                decisions["encode"][col] = f"Binary Mapping (0/1){rare_note}"
            elif nu <= 10:
                decisions["encode"][col] = f"One-Hot Encoding{rare_note}"
            elif nu <= 50:
                decisions["encode"][col] = f"Target or Frequency Encoding{rare_note}"
            else:
                decisions["encode"][col] = f"Hash Encoding or PCA-reduced Binary{rare_note}"
            continue

        # STEP 4: TRANSFORM (Rule 12: Explicit Numeric Rules)
        if col in risks["numeric_cols"]:
            is_skewed = col in skew_set
            has_outliers = col in outlier_set
            
            if is_skewed and has_outliers:
                decisions["transform"][col] = "Log-Transform + RobustScaler"
            elif is_skewed:
                decisions["transform"][col] = "Log / Power Transform"
            elif has_outliers:
                decisions["transform"][col] = "RobustScaler (IQR-based)"
            else:
                decisions["transform"][col] = "StandardScaler (Z-score)"
            continue

    # STEP 5: ENGINEER (Rule 9: Strict Gate)
    families = risks.get("feature_families", {})
    if families:
        # Only suggest if members haven't been dropped
        for stem, members in families.items():
            active_members = [m for m in members if m not in decisions["drop"]]
            if len(active_members) >= 2:
                eng_list = cast(list, decisions["engineer"])
                eng_list.append({
                    "family": stem,
                    "members": active_members,
                    "suggestion": f"Calculate delta/ratio among {stem} measurements"
                })
    
    if len(date_set) >= 2:
        eng_list = cast(list, decisions["engineer"])
        eng_list.append({
            "type": "temporal",
            "suggestion": "Calculate time intervals/diffs between date features"
        })

    return decisions


# ___________________________________________________________________________________________________________________________________________________________

def build_structured_dataset_profile(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Compile a complete, structured dataset profile including column-level facts.
    
    This acts as the master profiler function. It first calls `infer_data_risks` 
    to get the dataset-wide statistics. It then maps out explicit facts (like 
    missing counts, unique values, and exact dtypes) for every single column.

    Args:
        df (pd.DataFrame): The dataset to profile.
        target_column (Optional[str]): The column to evaluate as the target.

    Returns:
        Dict[str, Any]: A complete structured dictionary linking shape, risks, 
                        and per-column facts for the LLM context string.
    """
    risks = infer_data_risks(df, target_column=target_column)

    # Build exact column-level truth table so LLM never guesses
    column_facts = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        nunique = int(df[col].nunique(dropna=True))

        col_info: Dict[str, Any] = {
            "dtype": dtype,
            "missing_count": missing,
            "unique_values_count": nunique
        }

        # Sample unique values for categoricals to prevent hallucinated cardinality
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            samples = df[col].dropna().unique()
            col_info["example_values"] = [str(x) for x in samples[:5]]

        column_facts[col] = col_info

    # Build decisions based on the computed risks
    pipeline_decisions = generate_pipeline_decisions(df, risks, target_column=target_column)

    return {
        "dataset_scale": risks.get("size_summary", f"{df.shape[0]} rows"),
        "feature_counts": {
            "numeric": risks.get("numeric_feature_count", 0),
            "categorical": risks.get("categorical_feature_count", 0)
        },
        "feature_dimensionality": risks.get("dimensionality", "moderate"),
        "target_distribution": risks.get("target_distribution", {}),
        "missingness_summary": {
            "total_missing": risks.get("missing_total", 0),
            "missing_pct": risks.get("missing_pct", 0.0),
        },
        "pipeline_decisions": pipeline_decisions,
        "fully_missing_columns": risks.get("fully_missing_cols", []),
        "constant_columns": risks.get("constant_cols", []),
        "identifier_columns": risks.get("id_like_cols", []),
        "name_like_cols": risks.get("name_like_cols", []),
        "date_like_cols": risks.get("date_like_cols", []),
        "text_like_cols": risks.get("text_like_cols", []),
        "categorical_cols": risks.get("categorical_cols", []),
        "binary_categorical_cols": risks.get("binary_categorical_cols", []),
        "numeric_cols": risks.get("numeric_cols", []),
        "potential_leakage_cols": risks.get("potential_leakage_cols", []),
        "rare_category_summary": risks.get("rare_category_summary", {}),
        "feature_engineering_facts": {
            "skewed_columns": risks.get("skewed_numeric_cols", []),
            "outlier_columns": risks.get("outlier_numeric_cols", []),
            "correlation_clusters": risks.get("correlation_clusters", []),
            "rare_categories": risks.get("rare_category_summary", {}),
            "redundant_notes": risks.get("redundant_feature_notes", []),
            "feature_families": risks.get("feature_families", {})
        },
        "correlation_clusters": risks.get("correlation_clusters", []),
        "outlier_columns": risks.get("outlier_numeric_cols", []),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_risks": risks,
        "majority_class_label": risks.get("majority_class_label"),
        "majority_class_pct": risks.get("majority_class_pct"),
        "minority_class_label": risks.get("minority_class_label"),
        "minority_class_pct": risks.get("minority_class_pct"),
        "column_level_facts": column_facts
    }
