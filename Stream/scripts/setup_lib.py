import streamlit as st
import requests
import psutil
import json
import time
import os
from user_config import save_config, OLLAMA_HOST, is_setup_complete, load_config

# Constants for Recommendations
RECOMMENDED_DEFAULT = {
    "planner": "deepseek-r1:1.5b",
    "writer": "llama3.2:latest"
}

# Model Role Mapping for Validation/Suggestions
ROLE_SUGGESTIONS = {
    "planner": ["deepseek-r1:1.5b", "deepseek-r1:8b", "llama3.2:1b"],
    "writer": ["llama3.2:latest", "mistral:latest", "phi3:latest", "llama3.2:1b"]
}

# Model Metadata for UI
MODEL_INFO = {
    "deepseek-r1:1.5b": {"size_gb": 1.1, "ram_gb": 4, "type": "Reasoning"},
    "deepseek-r1:8b": {"size_gb": 4.7, "ram_gb": 16, "type": "Reasoning"},
    "llama3.2:latest": {"size_gb": 2.0, "ram_gb": 8, "type": "General"},
    "llama3.2:1b": {"size_gb": 1.3, "ram_gb": 4, "type": "General"},
    "mistral:latest": {"size_gb": 4.1, "ram_gb": 12, "type": "General"},
    "phi3:latest": {"size_gb": 2.3, "ram_gb": 4, "type": "General"},
}

def get_system_ram():
    return round(psutil.virtual_memory().total / (1024**3), 1)

def check_ollama():
    """Robust check for Ollama with specific status."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/", timeout=2)
        return True, "Ollama is running"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused (is Ollama serve running?)"
    except requests.exceptions.Timeout:
        return False, "Connection timed out"
    except Exception as e:
        return False, str(e)

def get_rich_models():
    """Fetch all models from Ollama with size info."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return {m["name"]: m for m in models}
    except:
        pass
    return {}

def pull_model_stream(model_name):
    """Pull an Ollama model and yield progress."""
    url = f"{OLLAMA_HOST}/api/pull"
    payload = {"name": model_name}
    try:
        with requests.post(url, json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    status = chunk.get("status", "Processing...")
                    progress = 0
                    if "completed" in chunk and "total" in chunk:
                        progress = chunk["completed"] / chunk["total"]
                    yield status, progress
    except Exception as e:
        yield f"Error: {str(e)}", 0

def render_setup_css():
    st.markdown("""
        <style>
        .stButton > button { width: 100%; border-radius: 10px; height: 3em; }
        .recommendation-card {
            background-color: rgba(40, 167, 69, 0.1);
            border: 1px solid #28A745;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .custom-card {
            background-color: rgba(30, 144, 255, 0.1);
            border: 1px solid #1E90FF;
            padding: 20px;
            border-radius: 15px;
        }
        .ram-badge {
            font-size: 0.8em;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: bold;
        }
        .ram-low { background-color: #28A745; color: white; }
        .ram-med { background-color: #FFC107; color: black; }
        .ram-high { background-color: #DC3545; color: white; }
        </style>
    """, unsafe_allow_html=True)

def show_connection_error(error_msg):
    st.title("🔌 Waiting for Ollama")
    st.error(f"**Status:** {error_msg}")
    
    st.info("Searching for Ollama on " + OLLAMA_HOST)
    
    with st.spinner("Auto-retrying in 3 seconds..."):
        st.markdown("""
        1. Open Ollama app or run `ollama serve` in terminal.
        2. Ensure port 11434 is open.
        """)
        time.sleep(3)
        st.rerun()

def show_setup_home():
    st.title("🚀 Welcome to Stream")
    ram = get_system_ram()
    
    st.markdown(f"Detected **{ram}GB RAM**.")
    
    # Logic for recommendations
    if ram < 8:
        rec_planner = "deepseek-r1:1.5b"
        rec_writer = "llama3.2:1b"
        advice = "💡 Optimized for your system (Low RAM)."
    elif ram < 16:
        rec_planner = "deepseek-r1:1.5b"
        rec_writer = "llama3.2:latest"
        advice = "💡 Balanced for your system."
    else:
        rec_planner = "deepseek-r1:8b"
        rec_writer = "llama3.2:latest"
        advice = "💡 High-performance setup."

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
        st.subheader("1-Click Recommended")
        st.markdown(f"**Planner:** {rec_planner}\n\n**Writer:** {rec_writer}")
        st.caption(advice)
        if st.button("Use Recommended Setup", type="primary"):
            st.session_state.planner_choice = rec_planner
            st.session_state.writer_choice = rec_writer
            st.session_state.setup_step = "download"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Advanced Customization")
        st.markdown("Pick any role for any model in your Ollama library.")
        if st.button("Customize Models"):
            st.session_state.setup_step = "customize"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def show_customization():
    st.title("⚙️ Custom Model Configuration")
    ram = get_system_ram()
    
    local_models = get_rich_models()
    model_list = sorted(list(set(list(local_models.keys()) + list(MODEL_INFO.keys()))))
    
    def get_model_label(m):
        if m in local_models:
            size = round(local_models[m]["size"] / (1024**3), 1)
            return f"{m} (Installed: {size}GB)"
        elif m in MODEL_INFO:
            return f"{m} (Pull required: {MODEL_INFO[m]['size_gb']}GB)"
        return m

    st.markdown("### Select Roles")
    
    col1, col2 = st.columns(2)
    with col1:
        planner = st.selectbox("Planner (Reasoning)", options=model_list, index=model_list.index("deepseek-r1:1.5b") if "deepseek-r1:1.5b" in model_list else 0, format_func=get_model_label)
        p_ram = MODEL_INFO.get(planner, {}).get("ram_gb", 4)
        st.caption(f"Suggested RAM: {p_ram}GB")
        
    with col2:
        writer = st.selectbox("Writer (Generative)", options=model_list, index=model_list.index("llama3.2:latest") if "llama3.2:latest" in model_list else 0, format_func=get_model_label)
        w_ram = MODEL_INFO.get(writer, {}).get("ram_gb", 8)
        st.caption(f"Suggested RAM: {w_ram}GB")

    total_ram = p_ram + w_ram
    if ram < total_ram:
        st.warning(f"⚠️ Warning: Total suggested RAM ({total_ram}GB) exceeds system ({ram}GB).")
    else:
        st.success(f"✅ System has enough RAM for this combo.")

    if st.button("Download & Save", type="primary"):
        st.session_state.planner_choice = planner
        st.session_state.writer_choice = writer
        st.session_state.setup_step = "download"
        st.rerun()
        
    if st.button("← Back"):
        st.session_state.setup_step = "home"
        st.rerun()

def show_download_progress():
    st.title("📥 Provisioning Brains")
    planner = st.session_state.planner_choice
    writer = st.session_state.writer_choice
    local_models = get_rich_models().keys()
    
    to_pull = []
    if planner not in local_models: to_pull.append(planner)
    if writer not in local_models and writer != planner: to_pull.append(writer)
    
    if not to_pull:
        save_config(planner, writer)
        st.session_state.setup_complete = True
        st.rerun()
        return

    for model in to_pull:
        st.markdown(f"#### Pulling **{model}**...")
        pbar = st.progress(0)
        status_txt = st.empty()
        for status, progress in pull_model_stream(model):
            status_txt.text(f"Status: {status}")
            pbar.progress(progress)
            
    st.success("All models ready!")
    save_config(planner, writer)
    st.session_state.setup_complete = True
    if st.button("Launch App"):
        st.rerun()

def boot_sequence():
    """Main entry point for app startup. Returns True if ready."""
    render_setup_css()
    
    # 1. Check Ollama
    is_up, msg = check_ollama()
    if not is_up:
        show_connection_error(msg)
        return False
        
    # 2. Check if setup is already perfectly complete (Zero-Click)
    if is_setup_complete():
        return True
        
    # 3. Check if required defaults are ALREADY installed (Silent Auto-Setup)
    local_models = get_rich_models().keys()
    if RECOMMENDED_DEFAULT["planner"] in local_models and RECOMMENDED_DEFAULT["writer"] in local_models:
        save_config(RECOMMENDED_DEFAULT["planner"], RECOMMENDED_DEFAULT["writer"])
        return True

    # 4. If we reach here, we need the UI
    if "setup_step" not in st.session_state:
        st.session_state.setup_step = "home"
        
    step = st.session_state.setup_step
    if step == "home":
        show_setup_home()
    elif step == "customize":
        show_customization()
    elif step == "download":
        show_download_progress()
        
    return False
