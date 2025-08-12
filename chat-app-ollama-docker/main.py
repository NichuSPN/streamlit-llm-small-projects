import streamlit as st
from openai import OpenAI

# --- Configuration ---
st.set_page_config(page_title="LLM Chat Interface", layout="wide")

PROVIDERS = {
    "ollama": {
        "name": "Ollama",
        "description": "Ollama is an open-source tool for running large language models locally.",
        "url": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    "docker": {
        "name": "Docker",
        "description": "Docker is a platform for developing, shipping, and running applications in containers.",
        "url": "http://localhost:12434/engines/v1",
        "api_key": "unused"
    }
}

# --- Model Discovery (runs once and is cached) ---
@st.cache_data
def discover_models():
    """Discovers available models from all configured providers."""
    models = {}
    errors = []
    for provider_key, provider_config in PROVIDERS.items():
        try:
            client = OpenAI(
                base_url=provider_config["url"],
                api_key=provider_config["api_key"]
            )
            for model in client.models.list():
                model_id = model.id
                models[model_id] = {
                    "name": model_id,
                    "provider": provider_key,
                    "display_name": f"{model_id} ({provider_config['name']})",
                }
        except Exception as e:
            errors.append(f"Could not connect to {provider_config['name']} at {provider_config['url']}: {e}")
    return models, errors

# --- Main App Logic ---
st.title("🤖 Unified LLM Chat")
st.write("Select a model from any running provider to start chatting.")

# Initialize session state for models and errors if they don't exist
if "models" not in st.session_state:
    with st.spinner("Discovering local models..."):
        st.session_state.models, st.session_state.errors = discover_models()

# Display any errors from model discovery
if st.session_state.errors:
    for error in st.session_state.errors:
        st.warning(error)

if not st.session_state.models:
    st.error("No models found from any provider. Please ensure Ollama or another provider is running and has models available.")
    st.stop()

# --- UI for Model Selection and Chat ---
models = st.session_state.models
model_options = {model_info["display_name"]: model_id for model_id, model_info in models.items()}

# Use a sidebar for model selection and other controls
with st.sidebar:
    st.header("Configuration")
    selected_display_name = st.selectbox(
        "Select a Model:",
        options=list(model_options.keys()),
        index=0,
        key="selected_model_display_name"
    )
    selected_model_id = model_options[selected_display_name]

    if st.button("Clear Chat History"):
        st.session_state[f"messages_{selected_model_id}"] = []
        st.rerun()

# Initialize chat history for the selected model if it doesn't exist
if f"messages_{selected_model_id}" not in st.session_state:
    st.session_state[f"messages_{selected_model_id}"] = []

# Display chat messages from history
for message in st.session_state[f"messages_{selected_model_id}"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input(f"Message {selected_model_id}..."):
    selected_provider_key = models[selected_model_id]["provider"]
    provider_config = PROVIDERS[selected_provider_key]

    client = OpenAI(base_url=provider_config["url"], api_key=provider_config["api_key"])

    st.session_state[f"messages_{selected_model_id}"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=selected_model_id,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state[f"messages_{selected_model_id}"]],
                stream=True,
            )
            response = st.write_stream(stream)
            st.session_state[f"messages_{selected_model_id}"].append({"role": "assistant", "content": response})
        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state[f"messages_{selected_model_id}"].append({"role": "assistant", "content": error_message})