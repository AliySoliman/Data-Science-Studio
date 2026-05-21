import os
import ollama
import streamlit as st
import re
import time
import threading
from typing import Optional
from .config import PLANNER_MODEL, WRITER_MODEL, OLLAMA_HOST

# ___________________________________________________________________________________________________________________________________________________________

# Initialize the Ollama client using the host provided in environment variables
# or defaulting to localhost (centralized in config.py)
client = ollama.Client(host=OLLAMA_HOST)

def _ollama_chat(model: str, system_prompt: str, user_prompt: str, 
                 max_tokens: int = 300, temperature: float = 0.7, _retry: bool = True) -> Optional[str]:
    """
    Low-level helper to query the Ollama API via the official Python client.
    
    This function handles the communication with the Ollama service. It enforces 
    sequential execution and optimized inference settings (limited tokens).
    If a model is missing (404), it attempts to pull and retry once.
    """
    try:
        start_time = time.time()
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            options={
                "num_predict": max_tokens,
                "temperature": temperature
            }
        )
        elapsed = time.time() - start_time
        print(f"DEBUG: Ollama execution ({model}): {elapsed:.2f}s")
        return response.get("message", {}).get("content", "").strip()
    except Exception as e:
        error_msg = str(e).lower()
        # Handle 404 Model Not Found
        if ("not found" in error_msg or "404" in error_msg) and _retry:
            print(f"DEBUG: 🚀 Model '{model}' missing during live request. Pulling now...")
            # Inform the user via Streamlit so the "Generate" button doesn't look stuck
            st.info(f"🚀 AI Model '{model}' not found. Pulling now... this may take a moment.")
            _ensure_model_exists(model)
            # Recursively retry once (set _retry=False to avoid infinite loops)
            return _ollama_chat(model, system_prompt, user_prompt, max_tokens, temperature, _retry=False)
            
        st.error(f"❌ Ollama API error ({model}): {str(e)}. Ensure Ollama is running.")
        return None

# ___________________________________________________________________________________________________________________________________________________________

def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> reasoning traces from model outputs.
    
    Models like `deepseek-r1` embed their chain-of-thought reasoning inside 
    <think> XML tags. This function strips those tags out so that only the 
    final, parsed answer is returned to the user or injected into the next prompt.

    Args:
        text (str): The raw output string from the language model.

    Returns:
        str: The cleaned string with all <think> blocks removed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ___________________________________________________________________________________________________________________________________________________________

def query_planner_llm(system_prompt: str, user_prompt: str, 
                      max_tokens: int = 300, temperature: float = 0.7) -> Optional[str]:
    """
    Send a reasoning/planning request to the PLANNER model.
    
    This function specifically targets the configured PLANNER_MODEL (e.g., deepseek-r1). 
    It also automatically strips out any reasoning traces (<think> tags) before 
    returning the final strategy.

    Args:
        system_prompt (str): The system instructions for the planner.
        user_prompt (str): The dataset facts and query for the planner.

    Returns:
        Optional[str]: The generated strategy string, or None if the call fails.
    """
    raw = _ollama_chat(PLANNER_MODEL, system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
    if raw:
        return _strip_think_tags(raw)
    return None

# ___________________________________________________________________________________________________________________________________________________________

def query_writer_llm(system_prompt: str, user_prompt: str, 
                     max_tokens: int = 300, temperature: float = 0.7) -> Optional[str]:
    """
    Send a prose/formatting request to the WRITER model.
    
    This function specifically targets the configured WRITER_MODEL (e.g., llama3.2). 
    It is used for generating the final, polished Markdown report based on the 
    planner's strategy.

    Args:
        system_prompt (str): The system instructions for the writer.
        user_prompt (str): The planner strategy and context facts.

    Returns:
        Optional[str]: The polished Markdown report, or None if the call fails.
    """
    return _ollama_chat(WRITER_MODEL, system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)


# Keep a thin compatibility shim so any future callers still work
query_llm = query_writer_llm

# ___________________________________________________________________________________________________________________________________________________________

def ask_brief(question: str, context: str, 
              max_tokens: int = 300, temperature: float = 0.7) -> str:
    """
    Writer-model helper for short 2-3 sentence plain-text answers.
    
    This function wraps the WRITER_MODEL call with a strict system prompt 
    enforcing concise, conversational answers based purely on the given context. 
    It is used for generating the executive summary and deployment risks.

    Args:
        question (str): The specific question to ask the model.
        context (str): The dataset facts provided as context.

    Returns:
        str: A brief 2-3 sentence response, or a fallback string.
    """
    system = (
        "You are a concise Senior Data Science advisor. "
        "Answer in 2-3 sentences using ONLY the provided context. "
        "Do not invent columns or facts."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return query_writer_llm(system, user, max_tokens=max_tokens, temperature=temperature) or "Not inferable from the supplied dataset profile."

# ___________________________________________________________________________________________________________________________________________________________

def ask_markdown(question: str, context: str, 
                 max_tokens: int = 600, temperature: float = 0.1) -> str:
    """
    Writer-model helper for structured Markdown output (e.g., tables).
    
    This function wraps the WRITER_MODEL call with a system prompt enforcing 
    exact Markdown formatting. It is primarily used for generating the structured 
    Model Selection and Feature Engineering tables.

    Args:
        question (str): The prompt detailing the required table format.
        context (str): The dataset facts and planner strategy.

    Returns:
        str: A Markdown-formatted string containing the table, or a fallback string.
    """
    system = (
        "You are a Senior Data Science advisor. "
        "Return ONLY valid markdown matching the requested format exactly. "
        "Use ONLY the provided context. Do not invent columns or facts."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return query_writer_llm(system, user, max_tokens=max_tokens, temperature=temperature) or "Not inferable from the supplied dataset profile."

# ___________________________________________________________________________________________________________________________________________________________

def _ensure_model_exists(model_name: str):
    """
    Check if a model exists on the Ollama host, and pull it if it's missing.
    """
    try:
        # Try to show model info
        client.show(model_name)
        print(f"DEBUG: ✅ Model {model_name} is available.")
    except Exception:
        # Pull model if missing
        print(f"DEBUG: 🚀 Pulling model {model_name} (automated provisioning)...")
        last_status = ""
        for part in client.pull(model=model_name, stream=True):
            status = part.get("status", "")
            if status != last_status:
                print(f"DEBUG: [{model_name}] {status}")
                last_status = status
        print(f"DEBUG: ✅ Model {model_name} successfully pulled.")

# ___________________________________________________________________________________________________________________________________________________________

def warmup_ollama():
    """
    Background warmup logic to ensure models are present and loaded into RAM.
    
    This verifies both models exist (pulling if missing) and then sends 
    a minimal prompt to hit the 'hot' state, eliminating first-load delay.
    """
    try:
        print(f"DEBUG: ⚡ Starting Ollama model warmup at {OLLAMA_HOST}...")
        
        # 1. Ensure models exist (Auto-pulling)
        _ensure_model_exists(PLANNER_MODEL)
        _ensure_model_exists(WRITER_MODEL)

        # 2. Hit models with minimal prompts to trigger load
        client.chat(model=PLANNER_MODEL, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
        client.chat(model=WRITER_MODEL, messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})
        
        print("DEBUG: ✅ Warmup complete. Both models in memory.")
    except Exception as e:
        # Non-critical: warmup failure shouldn't crash the main app, but we log it
        print(f"DEBUG: ⚠️ Warmup failed/aborted: {str(e)}")

# Launch warmup in a separate thread so it doesn't block app initialization
threading.Thread(target=warmup_ollama, daemon=True).start()
