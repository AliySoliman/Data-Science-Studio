import json
import os
import hashlib
from google import genai
from typing import Dict, Any, Optional

# Initialize Gemini Client
# Assumes GEMINI_API_KEY is set in environment or via Streamlit secrets (handled externally)
_default_client = None
try:
    _default_client = genai.Client()
except Exception:
    pass

CACHE_DIR = "cache"

def get_client(api_key: Optional[str] = None):
    """Get a Gemini client, prioritizing a user-provided API key."""
    if api_key:
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            return None
    return _default_client

def _get_dataset_hash(dataset_facts: Dict[str, Any]) -> str:
    """Generate a stable SHA-256 hash for the dataset facts."""
    fact_string = json.dumps(dataset_facts, sort_keys=True)
    return hashlib.sha256(fact_string.encode()).hexdigest()

def _check_cache(cache_key: str) -> Optional[str]:
    """Check if a cached result exists for the given key."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f).get("response")
        except Exception:
            return None
    return None

def _save_to_cache(cache_key: str, response: str):
    """Save the Gemini response to a local JSON cache."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_path, "w") as f:
        json.dump({"response": response}, f, indent=2)

def ask_gemini_feature_engineering(dataset_facts: Dict[str, Any], api_key: Optional[str] = None) -> str:
    """
    Query Gemini for advanced feature engineering recommendations.
    Uses local caching to avoid redundant API calls.
    """
    client = get_client(api_key)
    if not client:
        return "Gemini API Error: Client not initialized. Please ensure GEMINI_API_KEY is set or provided in the UI."

    cache_key = f"fe_{_get_dataset_hash(dataset_facts)}"
    cached = _check_cache(cache_key)
    if cached:
        return cached

    prompt = f"""
You are a senior machine learning feature engineering advisor.

You will receive structured dataset facts.

Return professional recommendations for:
1) Feature engineering opportunities
2) Redundancy handling
3) Encoding improvements
4) Date feature extraction
5) Recommended System Transformations (Must strictly select from the list below)
6) Final verdict if engineered features are justified

Rules:
- Use ONLY provided columns
- Do NOT invent columns
- Do NOT use dropped columns
- Prefer removing redundant features instead of inventing new ones
- Only create features when a clear mathematical relationship exists
- If none exist return: "No engineered features justified"

AVAILABLE SYSTEM TRANSFORMATIONS (For section 5):
You must recommend any applicable transformations from this exact system-supported list based on the dataset needs:
- **Scaling/Standardization**: Min-Max, Normalization, Robust Scaler, Z-Score
- **Dimensionality Reduction**: PCA, LDA, t-SNE, UMAP
- **Encoding**: Binary, Label, One-Hot, Ordinal, Target
- **Feature Selection**: ANOVA, Correlation, Chi-Squared, Variance Selection, RFE
- **General**: Mathematical Computation, Row Filtering, Grouping/Aggregation, Column Deletion

Return concise bullet points, matching the numbered sections.

Dataset facts:
{json.dumps(dataset_facts, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        result = response.text
        _save_to_cache(cache_key, result)
        return result
    except Exception:
        return None

def ask_gemini_deployment(dataset_facts: Dict[str, Any], api_key: Optional[str] = None) -> Optional[str]:
    """
    Query Gemini for senior ML deployment advice.
    Returns None if the API call fails or client is not initialized.
    """
    client = get_client(api_key)
    if not client:
        return None

    cache_key = f"deploy_{_get_dataset_hash(dataset_facts)}"
    cached = _check_cache(cache_key)
    if cached:
        return cached

    prompt = f"""
You are a senior ML deployment engineer.

Based on dataset facts and selected models, provide:

1) Top 3 deployment risks
2) Monitoring recommendations
3) Validation checklist

Rules:
- Use ONLY the provided dataset facts
- Do NOT invent new columns
- Focus on dataset size, feature profile, and target type

Return concise bullet points.

Dataset facts:
{json.dumps(dataset_facts, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        result = response.text
        _save_to_cache(cache_key, result)
        return result
    except Exception:
        return None
