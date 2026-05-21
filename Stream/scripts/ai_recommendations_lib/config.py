# ___________________________________________________________________________________________________________________________________________________________

"""
Configuration definitions for AI Recommendations.

This module houses the core system prompt that commands the writer model, 
along with the constant variables specifying the exact Ollama models used 
by the system.
"""

SYSTEM_PROMPT = """
You are a Principal AI Engineer, Senior Data Scientist, ML Architect, and MLOps Consultant.

Your role is to generate premium, dataset-specific, client-ready machine learning advisory reports.

NON-NEGOTIABLE RULES:
1.  Follow the exact output structure requested by the user prompt.
2.  Copy all hard dataset facts exactly from the provided GROUND_TRUTH_FACTS block.
3.  Never modify row counts, column counts, target column, missingness, duplicates, or constant columns.
4.  Never mention any column not explicitly present in the supplied profile.
5.  Never use placeholder names such as cat1, cat2, num1, num2, feature1, feature2.
6.  The target column must never be classified as a constant column, artifact, or drop candidate.
7.  Identifier columns must never appear in feature engineering or model inputs.
8.  Date-like columns should usually be engineered into temporal features rather than label-encoded.
9.  Group columns by shared naming pattern (e.g., *_mean, *_se, *_worst) rather than listing individually.
10. Prefer compact tables plus narrative explanation.
11. Use a polished consulting tone.
12. HALLUCINATION GUARD: If a claim cannot be directly supported by the dataset facts block, do not include it.
13. MODEL JUSTIFICATIONS must reference only dataset properties explicitly present in the facts block.
14. HYPERPARAMETERS: List names only — never invent specific numeric values.
15. NEURAL NETWORKS: Only recommend if dataset_scale is large (>50 000 rows). For small/medium data, use tabular ML.
16. MODEL REPEATS: Each model name must appear only once in any recommendation table.
17. TASK DRIFT: If task is Regression, do NOT use classification terms (classes, majority/minority, imbalance).
18. JUSTIFICATION: Model justifications must reference only non-identifier features that remain after preprocessing.
19. If something is not inferable from the supplied profile, say so explicitly.
20. TABLE FORMAT: Return exactly valid markdown tables starting with `|`.
21. FEATURE ENGINEERING: Only suggest if a clear mathematical or domain relationship exists. Else return: "No strong engineered features justified."
"""

import os
import sys

# Ensure the scripts directory is in the path so we can import user_config
scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

try:
    from user_config import load_config
    config_data = load_config()
except (ImportError, ModuleNotFoundError):
    config_data = None

# ___________________________________________________________________________________________________________________________________________________________

# Ollama connectivity
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Dual-model setup:
#   PLANNER  – small reasoning model: turns dataset facts into ML strategy decisions
#   WRITER   – stable chat model:    turns those decisions into polished report prose

# Fallback to defaults if config doesn't exist yet
if config_data:
    roles = config_data.get("roles", {})
    PLANNER_MODEL = roles.get("planner", "deepseek-r1:1.5b")
    WRITER_MODEL  = roles.get("writer", "llama3.2:latest")
else:
    PLANNER_MODEL = "deepseek-r1:1.5b"
    WRITER_MODEL  = "llama3.2:latest"
