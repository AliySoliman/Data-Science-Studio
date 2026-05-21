import streamlit as st
import pandas as pd
import time
from transformations.transformation_execution import execute_transformation
from models.models_execution import execute_model
from mapping_tables import TRANSFORM_STRATEGY_MAP, ENCODE_STRATEGY_MAP, MODEL_NAME_MAP, MODEL_DEFAULT_PARAMS, MODEL_REPORTING_MAP


def _fuzzy_lookup(category, strategy_map):
    """Try exact match first, then case-insensitive substring match."""
    if category in strategy_map:
        return strategy_map[category]
    cat_lower = category.lower()
    for key, val in strategy_map.items():
        if key.lower() == cat_lower:
            return val
    # substring: strategy_map key contained in category (e.g. "One-Hot Encoding" in "One-Hot Encoding (applied)")
    for key, val in strategy_map.items():
        if key.lower() in cat_lower or cat_lower in key.lower():
            return val
    return None


def execute_automation_pipeline(plan, df, pipeline_json, selected_models, progress_callback):
    """
    Sequentially executes the automation pipeline stages.
    """
    # ── MUST BE FIRST: Initialize all session state before any access ──────
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {}
    if 'transformations' not in st.session_state.pipeline:
        st.session_state.pipeline['transformations'] = []
    if 'ML' not in st.session_state.pipeline:
        st.session_state.pipeline['ML'] = []
    if 'report_items' not in st.session_state.pipeline:
        st.session_state.pipeline['report_items'] = []
    if 'selected_trans' not in st.session_state:
        st.session_state.selected_trans = []
    if 'load' not in st.session_state:
        st.session_state.load = False # Assume loaded if we are running the pipeline
    if 'df_original' not in st.session_state:
        st.session_state.df_original = df.copy() if df is not None else None
    if 'automation_progress' not in st.session_state:
        st.session_state.automation_progress = {"current_step": 0, "total_steps": 0, "message": "", "log": []}

    # ───────────────────────────────────────────────────────────────────────

    total_steps = len(plan)
    current_step = 0
    transformed_df = df.copy()

    def update_progress(msg, increment=True):
        nonlocal current_step
        if increment:
            current_step += 1
        progress_callback(current_step, total_steps, msg)
        time.sleep(0.5)

    # Stage A: Cleaning
    for clean_step in pipeline_json.get("cleaning", []):
        if clean_step["action"] == "drop_column":
            col = clean_step["column"]
            action_name = f"Drop column: {col}"
            update_progress(f"🧹 {action_name}")

            # singular 'column' key — required by apply_selected_transformations
            step_dict = {
                "name": action_name,
                "type": "transformation",
                "category": "delete",
                "column": col
            }
            st.session_state.pipeline["transformations"].append(step_dict)
            try:
                transformed_df = execute_transformation(
                    "transformation", "execution", {"df": transformed_df, "step": step_dict}
                )
                st.session_state.automation_progress["log"].append(f"✅ {action_name}")
            except Exception as e:
                st.session_state.automation_progress["log"].append(f"❌ Failed to drop {col}: {str(e)}")

        elif clean_step["action"] == "handle_missing":
            col = clean_step["column"]
            strategy = clean_step["strategy"]
            action_name = f"Handle missing: {col} ({strategy})"
            update_progress(f"🔧 {action_name}")

            # 'columns' list + 'strategies' dict — required by handle_missing_values
            step_dict = {
                "name": action_name,
                "type": "cleaning",
                "category": "Null handling",
                "columns": [col],
                "strategies": {col: strategy}
            }
            st.session_state.pipeline["transformations"].append(step_dict)
            try:
                transformed_df = execute_transformation(
                    "cleaning", "execution", {"df": transformed_df, "step": step_dict}
                )
                st.session_state.automation_progress["log"].append(f"✅ {action_name}")
            except Exception as e:
                st.session_state.automation_progress["log"].append(f"❌ Failed missing handling {col}: {str(e)}")

    # Stage B: Transformations (Scaling & Encoding)
    for trans_step in pipeline_json.get("transformations", []):
        t_type = trans_step["type"]
        category = trans_step["category"]
        cols = trans_step["columns"]

        mapped = None
        if t_type == "standardization":
            mapped = _fuzzy_lookup(category, TRANSFORM_STRATEGY_MAP)
        elif t_type == "encoding":
            mapped = _fuzzy_lookup(category, ENCODE_STRATEGY_MAP)

        if mapped:
            if mapped["category"] == "computation":
                # Special handling for computation-based transforms (like Log) which are 1-to-1
                for col in cols:
                    action_name = f"Log Transform: {col}"
                    update_progress(f"📐 {action_name}")

                    step_dict = {
                        "name": action_name,
                        "type": "transformation",
                        "category": "computation",
                        "expr": f"np.log([{col}] + 1e-9)", # Use numpy log with numerical stability
                        "new_column": col # Apply in-place by using same column name
                    }
                    st.session_state.pipeline["transformations"].append(step_dict)
                    try:
                        transformed_df = execute_transformation(
                            "transformation", "execution", {"df": transformed_df, "step": step_dict}
                        )
                        st.session_state.automation_progress["log"].append(f"✅ {action_name}")
                    except Exception as e:
                        st.session_state.automation_progress["log"].append(f"❌ Failed {action_name}: {str(e)}")
            else:
                # Standard standardization/encoding which handle multiple columns at once
                action_name = f"{mapped['category']} on {', '.join(cols[:2])}"
                if len(cols) > 2:
                    action_name += "..."
                update_progress(f"📐 {action_name}")

                step_dict = {
                    "name": action_name,
                    "type": mapped["type"],
                    "category": mapped["category"],
                    "columns": cols,
                    "edit_values": {}
                }
                st.session_state.pipeline["transformations"].append(step_dict)
                try:
                    transformed_df = execute_transformation(
                        mapped["type"], "execution", {"df": transformed_df, "step": step_dict}
                    )
                    st.session_state.automation_progress["log"].append(f"✅ {action_name}")
                except Exception as e:
                    st.session_state.automation_progress["log"].append(f"⚠️ Skipped {action_name}: {str(e)}")
        else:
            st.session_state.automation_progress["log"].append(
                f"⚠️ No mapping found for: {category} — skipped"
            )


    # Stage D: Model Training
    target = pipeline_json.get("target_column")
    task_type = pipeline_json.get("task_type", "")  # "Regression", "Classification", "Clustering"

    # Safety: if target not in df, skip model training entirely
    if target and target not in transformed_df.columns:
        st.session_state.automation_progress["log"].append(
            f"❌ Target column '{target}' not found in transformed data. Skipping model training."
        )
    else:
        all_features = [c for c in transformed_df.columns if c != target]
        # Numeric-only features (safe for regressors, clustering, distance-based models)
        numeric_features = transformed_df[all_features].select_dtypes(include=['number']).columns.tolist()

    for model_name_ai in selected_models:
        # Try exact match first, then fuzzy fallback
        model_key = MODEL_NAME_MAP.get(model_name_ai) or _fuzzy_lookup(model_name_ai, MODEL_NAME_MAP)
        if not model_key:
            st.session_state.automation_progress["log"].append(
                f"⚠️ Model mapping not found for: {model_name_ai}"
            )
            continue

        action_name = f"Train: {model_name_ai}"
        update_progress(f"🤖 {action_name}")

        params = MODEL_DEFAULT_PARAMS.get(model_key, {}).copy()

        # Merge manual_params to top-level for scripts that expect parameters flattened (e.g., Logistic Regression)
        if "manual_params" in params and isinstance(params["manual_params"], dict):
            for k, v in params["manual_params"].items():
                params.setdefault(k, v)

        # Ensure foundational model keys exist for execution scripts to avoid Missing Argument errors
        params.setdefault("use_grid_search", False)
        params.setdefault("cv_folds", 5)
        params.setdefault("param_grid", {})
        params.setdefault("manual_params", {})

        # Special case defaults for specific model types if missing
        params.setdefault("manual_k", 3)
        params.setdefault("random_state", 42)
        params.setdefault("alpha", 1.0)

        # Choose feature set: regression/distance models get numeric only; classifiers get all
        _regression_keys = {"Random Forest Regression", "KNN_Regressor", "Linear Regression",
                            "Gradient Boosting Regressor", "XGBoost Regressor"}
        _clustering_keys = {"KMeans Clustering", "DBSCAN", "Hierarchical Clustering", "Unsupervised KNN Clustering"}
        if model_key in _regression_keys or model_key in _clustering_keys:
            features_for_model = numeric_features
        else:
            features_for_model = all_features

        params.update({
            "df": transformed_df,
            "features": features_for_model,
            "target": target,
            "edit": False
        })

        try:
            results = execute_model(model_key, "script", params)

            if results is None:
                st.session_state.automation_progress["log"].append(
                    f"❌ {model_name_ai} training returned no results"
                )
                continue

            # All model scripts already append their entry to st.session_state.pipeline["ML"]
            # Enrich the last entry: model name = internal key for FUNCTION_MAP routing
            # model_label = human-readable AI name for the UI
            last_ml_entry = st.session_state.pipeline["ML"][-1]
            last_ml_entry["model name"] = model_key          # e.g. "LR", "KNN_Regressor"
            last_ml_entry["model_label"] = model_name_ai     # e.g. "Logistic Regression"
            last_ml_entry["transformations_list"] = [
                s["name"] for s in st.session_state.pipeline["transformations"]
            ]
            last_ml_entry["comments"] = (
                f"Automatically trained by AI Pipeline. Rating: "
                f"{next((m.get('rating') for m in pipeline_json.get('models', []) if m['name'] == model_name_ai), 'N/A')}"
            )

            st.session_state.automation_progress["log"].append(f"✅ {action_name}")

            # Stage E: Auto-Reporting
            # Retrieve model-specific reporting functions from mapping_tables
            # Default fallback choices
            default_choices = [
                {"function_name": "create_performance_metrics_plot", "user_title": "Performance Metrics"},
                {"function_name": "create_confusion_matrix_plot", "user_title": "Confusion Matrix"}
            ]
            
            # If regression, different default fallback
            if pipeline_json.get("task_type") == "Regression":
                default_choices = [
                    {"function_name": "create_performance_comparison", "user_title": "Performance Comparison"},
                    {"function_name": "create_residuals_plot", "user_title": "Residuals Plot"}
                ]

            report_item = {
                "id": int(time.time() * 1000),
                "type": "Machine Learning",
                # pipeline_step_name must equal the FUNCTION_MAP key in reports_execution.py
                "pipeline_step_name": model_key,
                # Human-readable label for expander header
                "pipeline_display_name": model_name_ai,
                "pipeline_source_index": len(st.session_state.pipeline["ML"]) - 1,
                "choices": MODEL_REPORTING_MAP.get(model_key, default_choices)
            }


            st.session_state.pipeline["report_items"].append(report_item)
            st.session_state.automation_progress["log"].append(
                f"✅ Report added for {model_name_ai}"
            )

        except Exception as e:
            st.session_state.automation_progress["log"].append(
                f"❌ Error training {model_name_ai}: {str(e)}"
            )

    update_progress("✨ Pipeline Complete!", increment=False)
    st.session_state.automation_done = True
    st.session_state.automation_running = False