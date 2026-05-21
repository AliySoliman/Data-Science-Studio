import pandas as pd
import re
from typing import Dict, Any, Optional
from .data_profiling import infer_task_type, build_structured_dataset_profile
from .llm_integration import query_planner_llm, ask_brief, ask_markdown
from .AI_gemini_advisor import ask_gemini_feature_engineering, ask_gemini_deployment
import streamlit as st

# ___________________________________________________________________________________________________________________________________________________________

def validate_hyperparameter_column(text: str) -> str:
    """
    Post-generation validator to fix common LLM mistakes 
    in the hyperparameter column.
    """
    if not text: return ""
    lines = text.split("\n")
    fixed = []
    HYPERPARAMS = {
        "Random Forest": "n_estimators, max_depth, min_samples_split",
        "Logistic Regression": "C, penalty, solver",
        "SVM": "C, kernel, gamma",
        "SVC": "C, kernel, gamma",
        "KNN": "n_neighbors, weights, metric",
        "Decision Tree": "max_depth, min_samples_split, criterion",
        "Linear Regression": "fit_intercept, normalize",
        "KMeans": "n_clusters, init, max_iter",
        "DBSCAN": "eps, min_samples, metric",
        "Hierarchical Clustering": "n_clusters, linkage, affinity",
        "XGBoost": "learning_rate, max_depth, n_estimators",
        "Gradient Boosting": "learning_rate, n_estimators, max_depth",
        "Naive Bayes": "var_smoothing"
    }
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 5:  # | Model | Why | Params | Strength |
                model_cell = parts[1].strip()
                param_cell = parts[3].strip()
                # If param cell is empty, None, or too short
                if not param_cell or param_cell.lower() in ("none", "-", "n/a", ""):
                    for model_name, default_params in HYPERPARAMS.items():
                        if model_name.lower() in model_cell.lower():
                            parts[3] = f" {default_params} "
                            break
                line = "|".join(parts)
        fixed.append(line)
    return "\n".join(fixed)

def build_mini_context(profile: Dict[str, Any], task_type: str) -> str:
    """
    Build a task-specific mini-context to reduce distraction for small LLMs.
    """
    risks = profile.get("target_risks", {})
    feature_counts = profile.get("feature_counts", {})
    h_p = risks.get("high_correlations", [])
    
    ctx = (
        f"Task: {task_type}. Scale: {profile.get('dataset_scale', 'medium')}. "
        f"Rows: {profile.get('row_count', 'unknown')}. "
        f"Features: {feature_counts.get('numeric', 0)} numeric, {feature_counts.get('categorical', 0)} categorical. "
        f"High corrs: {', '.join(map(str, h_p[:3])) if h_p else 'none'}. "
        f"Imbalance: {risks.get('imbalance_level', 'none')}."
    )
    return ctx

# ___________________________________________________________________________________________________________________________________________________________

def plan_ml_strategy(profile_ctx: str, task_type: str, dataset_scale: str = "medium") -> str:
    """
    Planner-model call: reason about ML strategy given Python-computed facts.
    
    This function sends the raw dataset profile to the reasoning planner LLM.
    The planner evaluates rules based on the dataset scale and task type, 
    and outputs a brief strategy covering top models, preprocessing needs, 
    and deployment concerns. This strategy is injected as grounding context 
    for all subsequent writer calls.

    Args:
        profile_ctx (str): The compressed summary of dataset facts.
        task_type (str): "Classification" or "Regression".
        dataset_scale (str): The rough size category (e.g. "small", "large").

    Returns:
        str: A concise strategy string mapping out the ML approach.
    """
    allowed_models = (
        "Logistic Regression, Random Forest Classifier, XGBoost Classifier, Gradient Boosting Classifier, KNN, SVM, Naive Bayes, Decision Tree"
        if task_type == "Classification"
        else "Linear Regression, Random Forest Regressor, XGBoost Regressor, Gradient Boosting Regressor, KNN Regressor"
    )
    nn_rule = (
        "Do NOT recommend neural networks — dataset_scale is not large (>50 000 rows)."
        if dataset_scale != "large"
        else "Neural networks may be considered given the large dataset scale."
    )
    system = (
        "You are an expert ML strategist.\n"
        f"ALLOWED MODELS: {allowed_models}.\n"
        f"{nn_rule}\n"
        "RULES:\n"
        "- Justify model choices using ONLY facts present in the profile (row count, dimensionality, correlations, imbalance).\n"
        "- Every model justification MUST reference at least ONE specific dataset fact.\n"
        "- Valid dataset facts include:\n"
        "  - row count\n"
        "  - feature dimensionality\n"
        "  - target distribution\n"
        "  - class imbalance\n"
        "  - correlation structure\n"
        "  - outliers\n"
        "  - missing values\n"
        "- Do not use generic phrases such as:\n"
        "  - good for tabular data\n"
        "  - works well in many cases\n"
        "  - powerful model\n"
        "  - popular choice\n"
        "- Each justification must explain why the model fits THIS dataset, not tabular data in general.\n"
        "- Do NOT repeat models in your strategy. One model per ranking slot.\n"
        "- If task is Regression, NEVER use classification terms (imbalance, classes).\n"
        "- If rows < 5000, mention that LightGBM/XGBoost may not provide large gains over Random Forest on this small dataset.\n"
        "- SVM RULE: SVM is highly effective for small datasets but may require careful kernel selection and scaling.\n"
        "- List HYPERPARAMETER NAMES ONLY — never specific numeric values.\n"
        "Output a compact strategy (under 200 words) with bullet points covering:\n"
        "1. Top 3 recommended models (ranked) — one-line justification citing dataset scale/structure.\n"
        "2. Top 2 preprocessing priorities (cite high correlations or skewness if present).\n"
        "3. Best evaluation metric and why (tie to imbalance_flag or task type).\n"
        "4. Single most dataset-specific deployment concern."
    )
    user = f"Task type: {task_type}\n\nDataset profile:\n{profile_ctx}"
    result = query_planner_llm(system, user)
    return result or "Strategy: not inferable from the supplied profile."

# ___________________________________________________________________________________________________________________________________________________________

def build_report_from_facts(
    df: pd.DataFrame,
    task_type: str = "Classification",
    target_column: Optional[str] = None,
    use_gemini: bool = False,
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    user_notes: Optional[str] = None
 ) -> str:
    """
    Python-first report assembly acting as the main pipeline coordinator.
    
    All facts are computed deterministically in Python (via data_profiling), 
    meaning the LLM only explains, summarizes, and justifies. The LLM never 
    decides task type, metrics, imbalance thresholds, or deployment structure. 
    This function orchestrates the interactions between the Planner and Writer
    models to assemble the final 5-section Advisory Report.

    Args:
        df (pd.DataFrame): The main dataset backing the report.
        task_type (str): The machine learning task type.
        target_column (Optional[str]): The column designated as the prediction target.
        use_gemini (bool): Whether to use Gemini for enhanced logic.
        api_key (Optional[str]): API key for Gemini.
        progress_callback (Optional[callable]): Callback for UI progress updates.
        user_notes (Optional[str]): Additional context/summary provided by the user.

    Returns:
        str: The final, polished Markdown report containing 5 distinct sections.
    """
    # ── #1 Auto-detect task type (Python, not LLM) ───────────────────────────
    task_type = infer_task_type(df, target_column)

    profile         = build_structured_dataset_profile(df, target_column=target_column)
    col_facts       = profile["column_level_facts"]
    
    # Extract from new clean top-level keys
    size_summary           = profile["dataset_scale"]
    feature_counts         = profile["feature_counts"]
    dimensionality         = profile["feature_dimensionality"]
    target_dist            = profile["target_distribution"]
    missing_summary        = profile["missingness_summary"]
    const_cols             = profile["constant_columns"]
    id_like                = profile["identifier_columns"]
    correlation_clusters   = list(profile.get("correlation_clusters", []))
    outliers               = list(profile.get("outlier_columns", []))
    fully_missing_cols     = list(profile.get("fully_missing_columns", []))
    
    # New facts
    name_cols              = list(profile.get("name_like_cols", []))
    date_cols              = list(profile.get("date_like_cols", []))
    text_cols              = list(profile.get("text_like_cols", []))
    leakage_cols           = list(profile.get("potential_leakage_cols", []))
    rare_categories        = dict(profile.get("rare_category_summary", {}))
    
    # --- New Deterministic Pipeline Decisions ---
    pipeline_decisions     = profile.get("pipeline_decisions", {})
    drop_decisions         = pipeline_decisions.get("drop", {})
    parse_decisions        = pipeline_decisions.get("parse", {})
    encode_decisions       = pipeline_decisions.get("encode", {})
    transform_decisions    = pipeline_decisions.get("transform", {})
    engineer_decisions     = pipeline_decisions.get("engineer", [])
    
    # Feature Engineering block
    fe_facts               = dict(profile.get("feature_engineering_facts", {}))
    skewed                 = list(fe_facts.get("skewed_columns", []))
    outliers               = list(fe_facts.get("outlier_columns", []))
    redundant_notes        = list(fe_facts.get("redundant_notes", []))

    # Pull remaining legacy or deep facts from risks if needed
    risks                  = profile["target_risks"]
    dataset_scale          = risks.get("dataset_scale", "medium")
    feature_families       = risks.get("feature_families", {})
    imbalance_flag         = risks.get("imbalance_flag", False)
    largest_cls_pct        = risks.get("largest_class_pct")
    high_corrs             = risks.get("high_correlations", [])

    total_miss = missing_summary["total_missing"]
    miss_pct   = missing_summary["missing_pct"]
    n_num_feat = feature_counts["numeric"]
    n_cat_feat = feature_counts["categorical"]

    # Feature-only views: exclude target and ID-like cols
    exclude_from_features = [c for c in ([target_column] + id_like) if c and c in df.columns]
    feature_df = df.drop(columns=exclude_from_features, errors="ignore")

    num_cols   = feature_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols   = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    skewed     = risks.get("skewed_numeric_cols", [])
    dup_rows   = risks["duplicates"]

    # Cardinality aliases (for later code)
    lc_cats    = risks.get("low_cardinality_cats", [])
    mc_cats    = risks.get("medium_cardinality_cats", [])
    hc_cats    = risks.get("high_cardinality_cats", [])

    if target_column and target_column in df.columns:
        target_dtype  = str(df[target_column].dtype)
        target_unique = int(df[target_column].nunique(dropna=True))
        target_note   = (
            f"Target: `{target_column}` ({target_dtype}, {target_unique} unique values) "
            f"— Task auto-detected as **{task_type}**"
        )
    else:
        target_note = f"No target column selected — Task: **{task_type}**"

    # ── Build rich context string for LLM calls ───────────────────────────────
    corr_preview = []
    for c in list(correlation_clusters):
        if len(corr_preview) >= 5: break
        corr_preview.append(c)
        
    outlier_preview = []
    for o in list(outliers):
        if len(outlier_preview) >= 10: break
        outlier_preview.append(o)
    
    dist_items = list(target_dist.items())
    dist_preview = []
    for k, v in dist_items:
        if len(dist_preview) >= 6: break
        dist_preview.append(f"{k}: {v}%")
    dist_str = ", ".join(dist_preview) if dist_preview else "N/A"
    
    fam_items = list(feature_families.items())
    fam_preview = []
    for stem, m in fam_items:
        if len(fam_preview) >= 5: break
        fam_preview.append(f"{stem} ({len(m)} cols)")
    families_str = "; ".join(fam_preview) if fam_preview else "None detected"
    
    short_ctx = (
        f"Task: {task_type}. Scale: {size_summary}. "
        f"Dim: {dimensionality} ({n_num_feat} num, {n_cat_feat} cat). "
        f"{target_note}. "
        f"Missing: {total_miss} ({miss_pct}%). Duplicates: {dup_rows}. "
        f"Correlations: {corr_preview}. "
        f"Outliers: {outlier_preview}. "
        f"Names: {name_cols}. Dates: {date_cols}. Text: {text_cols}. "
        f"Leakage: {leakage_cols}. "
        f"Rare cats: {rare_categories}. "
        f"IDs (excluded): {id_like}. "
        f"Families: {families_str}. "
        f"Distribution: {dist_str}. "
        f"Imbalance: {risks.get('imbalance_level', 'none')}."
    )

    if user_notes:
        short_ctx += f"\n\nUSER NOTES/SUMMARY:\n{user_notes}"

    # ── Planner pass ─────────────────────────────────────────────────────────
    if progress_callback: progress_callback("🧠 Reasoning about ML strategy (Planner)...")
    planner_strategy = str(plan_ml_strategy(short_ctx, task_type, dataset_scale))
    short_ctx = short_ctx + f"\n\nPLANNER STRATEGY (use to inform writing):\n{planner_strategy}"

    quality_rows = []
    quality_rows.append(f"📊 **Dataset Scale** — {size_summary.capitalize()}\n")
    quality_rows.append(f"📂 **Feature Count** — {n_num_feat} numeric, {n_cat_feat} categorical ({dimensionality})\n")

    if high_corrs:
        h_preview = []
        for h in sorted(list(high_corrs)):
            if len(h_preview) >= 3: break
            h_preview.append(str(h))
        quality_rows.append(f"⚠️ **High Correlations** — {', '.join(h_preview)}. Consider dropping one per pair to reduce redundancy.\n")

    # Grouped Correlation Clusters (#5 cleaning)
    if correlation_clusters:
        quality_rows.append("**Highly Correlated Feature Clusters Detected:**")
        for i, cluster in enumerate(list(correlation_clusters)):
            cluster_list = sorted(list(cluster))
            c_p = []
            for item in cluster_list:
                if len(c_p) >= 6: break
                c_p.append(str(item))
            preview_str = ", ".join(c_p)
            if len(cluster_list) > 6:
                preview_str += f" and {len(cluster_list) - 6} more"
            quality_rows.append(f"  - **Cluster {i+1}:** {preview_str}")
        quality_rows.append("")

    if total_miss == 0:
        quality_rows.append("✅ **Missing Values** — None detected. Dataset is complete.\n")
    else:
        missing_cols = [(c, v["missing_count"]) for c, v in col_facts.items() if v["missing_count"] > 0]
        # Skip fully missing from this list as they are handled in quality_rows later or summarized
        m_preview = []
        for pair in list(missing_cols):
            if len(m_preview) >= 6: break
            m_preview.append(pair)
            
        for col, cnt in m_preview:
            pct = round(cnt / df.shape[0] * 100, 1)
            quality_rows.append(f"⚠️ **Missing: `{col}`** — {cnt} values ({pct}%)\n")

    # Filtered lists to avoid duplication
    m_set = set(fully_missing_cols)
    
    if dup_rows > 0:
        quality_rows.append(f"⚠️ **Duplicates** — {dup_rows} duplicate rows detected.\n")
    else:
        quality_rows.append("✅ **Duplicates** — None detected.\n")

    if outliers:
        o_clean = [o for o in outliers if o not in m_set]
        if o_clean:
            o_p = []
            for c in o_clean:
                if len(o_p) >= 8: break
                o_p.append(str(c))
            quality_rows.append(f"⚠️ **Outliers** — Detected in: `{'`, `'.join(o_p)}`" + (" ...\n" if len(o_clean) > 8 else "\n"))
    else:
        quality_rows.append("✅ **Outliers** — None detected above 5% IQR threshold.\n")

    if const_cols:
        quality_rows.append(f"🔴 **Constant Columns** — Drop: `{'`, `'.join(sorted(const_cols))}`\n")

    if id_like:
        id_f = [c for c in id_like if c not in m_set]
        if id_f:
            id_f_str = [str(c) for c in id_f]
            quality_rows.append(f"ℹ️ **Identifier Columns** (excluded from features): `{'`, `'.join(id_f_str)}`\n")

    if hc_cats:
        hc_f = [c for c in hc_cats if c not in m_set]
        if hc_f:
            hc_p = []
            for c in hc_f:
                if len(hc_p) >= 4: break
                hc_p.append(str(c))
            quality_rows.append(f"⚠️ **High-Cardinality Text** — `{'`, `'.join(hc_p)}`. "
                                "These are likely identifiers or personal attributes and may not contain predictive signal.\n")

    if imbalance_flag and task_type == "Classification":
        mj_label = str(profile.get('majority_class_label', 'Unknown'))
        mj_pct = str(profile.get('majority_class_pct', '0'))
        mn_label = str(profile.get('minority_class_label', 'Unknown'))
        mn_pct = str(profile.get('minority_class_pct', '0'))
        im_lvl = str(risks.get('imbalance_level', 'none')).upper()
        
        quality_rows.append(f"⚠️ **Target Imbalance** — Level: {im_lvl}. "
                            f"Majority: {mj_label} ({mj_pct}%) | "
                            f"Minority: {mn_label} ({mn_pct}%)\n")
    elif target_dist:
        quality_rows.append(f"✅ **Target Distribution** — Balanced ({dist_str})\n")

    if leakage_cols:
        l_preview = sorted(list(leakage_cols))
        l_p = []
        for x in l_preview:
            if len(l_p) >= 3: break
            l_p.append(str(x))
        l_str = ", ".join(l_p)
        quality_rows.append(f"🚨 **Potential Leakage** — `{l_str}` show extreme correlation (>0.95) with target. Recommend removal.\n")
        
    if rare_categories:
        r_cols = sorted(list(rare_categories.keys()))
        r_p = []
        for x in r_cols:
            if len(r_p) >= 3: break
            r_p.append(str(x))
        r_str = ", ".join(r_p)
        quality_rows.append(f"⚠️ **Rare Categories** — Detected in `{r_str}`. May cause instability if not handled.\n")

    if date_cols:
         d_preview = sorted(list(date_cols))
         d_p = []
         for x in d_preview:
             if len(d_p) >= 3: break
             d_p.append(str(x))
         d_str = ", ".join(d_p)
         quality_rows.append(f"📅 **Date Columns** — Detected: `{d_str}`. Ensure proper temporal splitting.\n")
         
    if text_cols:
         t_preview = sorted(list(text_cols))
         t_p = []
         for x in t_preview:
             if len(t_p) >= 3: break
             t_p.append(str(x))
         t_str = ", ".join(t_p)
         quality_rows.append(f"📝 **Natural Language** — `{t_str}` likely contain free text. Use NLP vectorization.\n")
    

    if progress_callback: progress_callback("✍️ Writing Executive Summary & Quality Assessment...")
    readiness = ask_brief(
        f"In 2 sentences, assess whether this {task_type} ({dataset_scale} dataset, "
        f"{df.shape[0]} rows, {n_num_feat} numeric + {n_cat_feat} categorical features) "
        f"is ready for ML and what the main concern is. Note: {target_note}.",
        short_ctx
    )

    # --- Section 2: Feature Engineering & Column Pipeline ---
    fe_lines = [
        "This section follows a formal 5-step pipeline to handle column-level preprocessing deterministically.\n"
    ]

    # --- Step 1: Pruning & Column Selection (Dropping) ---
    fe_lines.append("### Step 1: Pruning & Column Selection (Dropping)")
    if drop_decisions:
        for col, reason in sorted(drop_decisions.items()):
            fe_lines.append(f"  - **Drop `{col}`**: {reason}")
    else:
        fe_lines.append("✅ No columns identified for immediate pruning.")
    fe_lines.append("")

    # --- Step 2: Specialized Data Parsing ---
    fe_lines.append("### Step 2: Specialized Data Parsing")
    if parse_decisions:
        for col, action in sorted(parse_decisions.items()):
            fe_lines.append(f"  - **Parse `{col}`**: {action}")
    else:
        fe_lines.append("✅ No specialized parsing required.")
    fe_lines.append("")

    # --- Step 3: Categorical Encoding Strategy ---
    fe_lines.append("### Step 3: Categorical Encoding Strategy")
    if encode_decisions:
        for col, strategy in sorted(encode_decisions.items()):
            fe_lines.append(f"  - **Encode `{col}`**: {strategy}")
    else:
        fe_lines.append("✅ No categorical features requiring encoding.")
    fe_lines.append("")

    # --- Step 4: Numeric Transformation & Scaling ---
    fe_lines.append("### Step 4: Numeric Transformation & Scaling")
    if transform_decisions:
        # Group by strategy for cleaner output
        strat_map: Dict[str, list[str]] = {}
        for col, strat in transform_decisions.items():
            strat_map.setdefault(strat, []).append(str(col))
        
        for strat, cols in sorted(strat_map.items()):
            # Use a loop instead of slicing to satisfy picky type checkers
            c_p = []
            for i, c in enumerate(sorted(cols)):
                if i >= 15: break
                c_p.append(f"`{c}`")
            c_str = ", ".join(c_p)
            if len(cols) > 15:
                c_str += f" and {len(cols)-15} more"
            fe_lines.append(f"- **{strat}**:\n  - {c_str}")
        fe_lines.append("\n> 🌲 **Note:** Tree-based models are robust to these distributions and don't strictly require scaling.")
    else:
        fe_lines.append("✅ No numeric features requiring transformation.")
    fe_lines.append("")

    # ── #15/#16/#17 Section 3: Model Selection ───────────────────────────────
    allowed = (
        "Logistic Regression, Random Forest Classifier, XGBoost Classifier, Gradient Boosting Classifier, KNN, SVM, Naive Bayes, Decision Tree"
        if task_type == "Classification"
        else "Linear Regression, Random Forest Regressor, XGBoost Regressor, Gradient Boosting Regressor, KNN Regressor"
        if task_type == "Regression"
        else "KMeans, DBSCAN, Hierarchical Clustering, Unsupervised KNN"
    )
    baseline = "Logistic Regression" if task_type == "Classification" else "Linear Regression"
    nn_rule  = (
        "Do NOT recommend neural networks — the dataset is small/medium (<50 000 rows)."
        if dataset_scale != "large"
        else "Neural networks may be considered given the large dataset scale."
    )

    h_p = []
    if high_corrs:
        h_sorted = sorted(list(high_corrs))
        for x in h_sorted:
            if len(h_p) >= 3: break
            h_p.append(str(x))

    model_table_prompt = (
        f"Return ONLY this markdown table for {task_type}, no other text:\n\n"
        f"| Model | Why it fits | Key Hyperparameters | Strength |\n"
        f"|---|---|---|---|\n"
        f"| Random Forest {'Classifier' if task_type == 'Classification' else 'Regressor'} | [cite a dataset fact] | n_estimators, max_depth, min_samples_split | Champion |\n"
        f"| {baseline} | [cite a dataset fact] | {'C, penalty, solver' if task_type == 'Classification' else 'fit_intercept, normalize'} | Baseline |\n"
        f"| {'SVC' if task_type == 'Classification' else 'KNN Regressor'} | [cite a dataset fact] | {'C, kernel, gamma' if task_type == 'Classification' else 'n_neighbors, weights'} | Strong Alternative |\n"
        f"| {'XGBoost Classifier' if task_type == 'Classification' else 'XGBoost Regressor'} | [cite a dataset fact] | learning_rate, max_depth, n_estimators | Experimental |\n"
        f"| {'Decision Tree' if task_type == 'Classification' else 'Gradient Boosting Regressor'} | [cite a dataset fact] | max_depth, min_samples_split | Experimental |\n\n"
        f"Rules: ONLY use these model names: {allowed}. "
        f"Strength must be: Champion, Strong Alternative, Baseline, Experimental. "
        f"Each 'Why it fits' must mention row count, correlation, or imbalance. No empty cells."
    )
    if progress_callback: progress_callback("🤖 Generating Targeted Model Selection Table...")
    
    # Use targeted mini-context for the table generation to avoid distraction
    table_ctx = build_mini_context(profile, task_type)
    model_section = ask_markdown(model_table_prompt, table_ctx)
    if model_section:
        model_section = validate_hyperparameter_column(model_section)

    # ── Gemini Advisor Pass ──────────────────────────────────────────────
    # Build the Dataset Facts Payload for Gemini
    candidate_models = []
    if model_section:
        candidate_models = [line.split('|')[1].strip() for line in model_section.split('\n') 
                          if '|' in line and not any(x in line for x in ['Model', '---', 'Expected'])]

    dataset_facts = {
        "task_type": task_type,
        "target": target_column,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "drop_columns": list(drop_decisions.keys()),
        "missing_columns": [c for c, v in col_facts.items() if v["missing_count"] > 0],
        "outlier_columns": outliers,
        "correlation_clusters": [list(c) for c in correlation_clusters],
        "rare_categories": rare_categories,
        "candidate_models": candidate_models
    }

    fe_lines.append("### Step 5: Engineered Features Justification")
    fe_rationale = None
    if use_gemini:
        fe_rationale = ask_gemini_feature_engineering(dataset_facts, api_key=api_key)
        if fe_rationale:
            fe_lines.append("### 🏆 AI-Powered Feature Engineering Recommendations (Gemini)")
        else:
            # Silent fallback if Gemini fails
            use_gemini = False 

    if not use_gemini or not fe_rationale:
        fe_lines.append("### 🤖 Local AI Feature Engineering Recommendations")
        fe_prompt = (
            "Based on the dataset profile and identified correlations/outliers, "
            "recommend 2-3 specific feature engineering steps (e.g. interaction terms, "
            "log transforms, or specific date extractions).\n"
            "You MUST also include a distinct section titled 'Recommended System Transformations' "
            "where you recommend specific techniques from our AVAILABLE SYSTEM TRANSFORMATIONS based on the dataset needs.\n\n"
            "AVAILABLE SYSTEM TRANSFORMATIONS:\n"
            "- Scaling/Standardization: Min-Max, Normalization, Robust Scaler, Z-Score\n"
            "- Dimensionality Reduction: PCA, LDA, t-SNE, UMAP\n"
            "- Encoding: Binary, Label, One-Hot, Ordinal, Target\n"
            "- Feature Selection: ANOVA, Correlation, Chi-Squared, Variance Selection, RFE\n"
            "- General: Mathematical Computation, Row Filtering, Grouping/Aggregation, Column Deletion\n\n"
            "Keep it concise and technical.\n"
            "IMPORTANT: Do NOT write any Python code, scripts, or formulas in your response. "
            "Provide ONLY the conceptual and mathematical explanation of the changes."
        )
        if progress_callback: progress_callback("🔧 Crafting Feature Engineering recommendations...")
        fe_rationale = ask_markdown(fe_prompt, short_ctx)
    
    fe_lines.append(fe_rationale)

    fe_text = "\n".join(fe_lines)

    # ── #18/#19 Section 4: Evaluation — purely Python ────────────────────────
    if task_type == "Classification":
        if imbalance_flag:
            primary_metric = "**AUC-ROC** + **F1-Score** (class imbalance detected — accuracy alone is misleading)"
        else:
            primary_metric = "**AUC-ROC** + **F1-Score** + **Accuracy**"
        val_strategy = "**Stratified K-Fold (k=5)** — preserves class distribution in every fold"
        secondary_metrics = "Precision, Recall, Confusion Matrix"
    else:
        primary_metric   = "**RMSE** (penalises large errors) + **MAE** + **R²** (variance explained)"
        val_strategy     = "**K-Fold (k=5)** — standard for regression tasks"
        secondary_metrics = "Residual plots, feature importance"

    # ── Deployment Advisor Pass ──────────────────────────────────────────
    deploy_commentary = None
    if use_gemini:
        deploy_commentary = ask_gemini_deployment(dataset_facts, api_key=api_key)
    
    if not deploy_commentary:
        deploy_prompt = (
            "Provide 3 concise deployment risks and 2 monitoring recommendations "
            "specific to this dataset and the recommended models."
        )
        if progress_callback: progress_callback("🚀 Finalizing Production & Deployment Strategy...")
        deploy_commentary = ask_brief(deploy_prompt, short_ctx)

    lc_mc_p = []
    lc_mc_src = list(lc_cats + mc_cats)
    for x in lc_mc_src:
        if len(lc_mc_p) >= 2: break
        lc_mc_p.append(str(x))
    lc_mc_p_str = ", ".join(lc_mc_p) if lc_mc_p else "N/A"

    eval_extra = ""
    if imbalance_flag and task_type == "Classification":
        eval_extra = (f"- **Imbalance Handling:** Class weighting or oversampling (SMOTE) — "
                      f"largest class is {largest_cls_pct}% of target\n")
    
    subgroup_note = f"- **Subgroup Monitoring:** Track metrics across {lc_mc_p_str}\n\n"

    # ── Assemble report ───────────────────────────────────────────────────────
    report = (
        f"## 1️⃣ Executive Summary & Data Quality Assessment 📊\n\n"
        f"> {target_note}\n\n"
        f"> {readiness}\n\n"
        f"**Computed Data Quality Facts:**\n\n"
        f"{chr(10).join(quality_rows)}\n\n"
        f"---\n\n"
        f"## 2️⃣ Advanced Feature Engineering & Transformations 🔧\n\n"
        f"{chr(10).join(fe_lines)}\n\n"
        f"---\n\n"
        f"## 3️⃣ Targeted Model Selection 🤖\n\n"
        f"{model_section}\n\n"
        f"---\n\n"
        f"## 4️⃣ Evaluation & Validation Framework 📈\n\n"
        f"- **Validation Design:** {val_strategy}\n"
        f"- **Primary Metrics:** {primary_metric}\n"
        f"- **Secondary Metrics:** {secondary_metrics}\n"
        f"{eval_extra}"
        f"{subgroup_note}"
        f"---\n\n"
        f"## 5️⃣ Production & Deployment Strategy 🚀\n\n"
        f"{deploy_commentary}\n\n"
        f"**Pipeline Reproducibility Checklist:**\n"
        f"- 💾 Save preprocessing pipeline (scaler, encoder) with `joblib` or `pickle`\n"
        f"- 📋 Define and validate input schema before inference\n"
        f"- 🔖 Version the model and training dataset (e.g., MLflow or DVC)\n"
        f"- 📊 Log prediction distributions in production for drift detection\n"
    )

    # ── #20 Extract structured JSON for automation ──────────────────────────
    pipeline_json = extract_pipeline_json(
        profile, task_type, target_column, model_section, pipeline_decisions
    )
    st.session_state.pipeline_json = pipeline_json

    return report

def extract_pipeline_json(profile, task_type, target_column, model_section, pipeline_decisions):
    """
    Constructs a structured dictionary from Python-computed facts and 
    parsed model recommendations to guide the automation executor.
    """
    # 1. Cleaning steps (from drop and missing)
    cleaning = []
    drop_decisions = pipeline_decisions.get("drop", {})
    for col, reason in drop_decisions.items():
        cleaning.append({"action": "drop_column", "column": col, "reason": reason})
    
    # Missing value strategies (extracted from column facts or pipeline decisions)
    col_facts = profile.get("column_level_facts", {})
    for col, facts in col_facts.items():
        if facts.get("missing_count", 0) > 0:
            # Simple heuristic mapping for the demo/automation layer
            if facts["missing_count"] / profile.get("row_count", 1) > 0.4:
                strategy = "drop_column"
            else:
                strategy = "median" if facts.get("is_numeric") else "mode"
            cleaning.append({"action": "handle_missing", "column": col, "strategy": strategy})

    # 2. Transformations (Scaling and Encoding)
    transformations = []
    
    # Numeric transforms
    trans_decisions = pipeline_decisions.get("transform", {})
    strat_map = {}
    for col, strat in trans_decisions.items():
        strat_map.setdefault(strat, []).append(col)
    
    for strat, cols in strat_map.items():
        transformations.append({
            "type": "standardization",
            "category": strat,
            "columns": cols
        })

    # Encoding
    encode_decisions = pipeline_decisions.get("encode", {})
    enc_strat_map = {}
    for col, strat in encode_decisions.items():
        enc_strat_map.setdefault(strat, []).append(col)
    
    for strat, cols in enc_strat_map.items():
        transformations.append({
            "type": "encoding",
            "category": strat,
            "columns": cols
        })

    # 3. Model parsing from the markdown table
    models = []
    if model_section:
        lines = model_section.split("\n")
        for line in lines:
            if "|" in line and "Strength" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    # | Model | Why | Params | Strength |
                    models.append({
                        "name": parts[0],
                        "rating": parts[3]
                    })

    return {
        "task_type": task_type,
        "target_column": target_column,
        "cleaning": cleaning,
        "transformations": transformations,
        "models": models
    }


# ___________________________________________________________________________________________________________________________________________________________

def generate_enhanced_recommendations(
    summary: str,
    df: pd.DataFrame,
    task_type: str = "Classification",
    target_column: Optional[str] = None,
    use_gemini: bool = False,
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None
 ) -> Optional[str]:
    """
    High-level entry point bridging the Streamlit UI and the report builder.
    
    This function acts as a wrapper for `build_report_from_facts`. Note that 
    the `task_type` passed from the UI might be ignored internally if the 
    builder auto-detects a cleaner task representation.

    Args:
        summary (str): (Legacy/Unused) Any free-form summary text string.
        df (pd.DataFrame): The active dataset.
        task_type (str): The suspected task type from the UI.
        target_column (Optional[str]): The targeted column.

    Returns:
        Optional[str]: The generated Markdown string report, or None if failed.
    """
    return build_report_from_facts(
        df, 
        task_type, 
        target_column=target_column, 
        use_gemini=use_gemini, 
        api_key=api_key,
        progress_callback=progress_callback,
        user_notes=summary
    )
