import json
import os
import requests

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "save files", "user_config.json")

# Centralized host detection (Env var OLLAMA_HOST or localhost)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

DEFAULT_CONFIG = {
    "roles": {
        "planner": "deepseek-r1:1.5b",
        "writer": "llama3.2:latest"
    }
}

def load_config():
    """Load the user configuration from disk."""
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            # Migration check for old schema
            if "roles" not in data:
                return {
                    "roles": {
                        "planner": data.get("planner_model", DEFAULT_CONFIG["roles"]["planner"]),
                        "writer": data.get("writer_model", DEFAULT_CONFIG["roles"]["writer"])
                    }
                }
            return data
    except Exception:
        return None

def save_config(planner_model, writer_model):
    """Save the user configuration to disk in the new roles schema."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    config = {
        "roles": {
            "planner": planner_model,
            "writer": writer_model
        }
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    return config

def get_installed_models():
    """Check Ollama for installed models."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except:
        pass
    return []

def is_setup_complete():
    """Check if config exists AND required models are pulled."""
    config = load_config()
    if not config:
        return False
    
    installed = get_installed_models()
    roles = config.get("roles", {})
    
    planner = roles.get("planner")
    writer = roles.get("writer")
    
    if not planner or not writer:
        return False
        
    return planner in installed and writer in installed
