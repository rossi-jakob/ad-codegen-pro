"""
Offline AI Code Generator — Streaming ChatGPT Style
Run:
    streamlit run app.py
"""

import os
import streamlit as st

from config import OUTPUT_DIR
from model_loader import ModelLoader
from rag_engine import RAGEngine
from code_generator import CodeGenerator
from project_generator import ProjectGenerator


# ─────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Offline AI Code Generator", page_icon="🤖")
st.title("🤖 Offline AI Code Generator")
st.caption("CodeLlama · Fully Offline · RAG-Enhanced · Streaming Enabled")


# ─────────────────────────────────────────────
# Cached System Loader
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_system():
    loader = ModelLoader()
    loader.load()

    rag = RAGEngine()
    rag.initialize()

    generator = CodeGenerator(loader, rag)
    project_gen = ProjectGenerator()

    return loader, rag, generator, project_gen


# ─────────────────────────────────────────────
# Safe Session Initialization
# ─────────────────────────────────────────────
if "initialized" not in st.session_state:

    loader, rag, generator, project_gen = load_system()

    st.session_state.loader = loader
    st.session_state.rag = rag
    st.session_state.generator = generator
    st.session_state.project_gen = project_gen
    st.session_state.messages = []
    st.session_state.initialized = True


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────
with st.sidebar:

    st.header("Controls")

    if st.button("🧹 Clear Conversation"):
        st.session_state.generator.conversation_history.clear()
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Commands")
    st.markdown(
        """
        `/project <desc>` — Generate full project  
        `/add <file>` — Add file to knowledge base  
        `/clear` — Clear conversation  
        `/quit` — Reload page  
        """
    )


# ─────────────────────────────────────────────
# Display Chat History
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
# Project Handler
# ─────────────────────────────────────────────
def handle_project(description: str):

    generator = st.session_state.generator
    project_gen = st.session_state.project_gen

    with st.spinner("Generating project..."):
        files = generator.generate_project(description)

    response_text = ""

    for filepath, content in files.items():
        response_text += (
            f"\n### 📄 {filepath}\n"
            f"```{filepath.split('.')[-1]}\n"
            f"{content}\n```\n"
        )

    project_name = "_".join(description.split()[:3]).lower()
    project_gen.save_project(project_name, files)

    return response_text


# ─────────────────────────────────────────────
# Add File Handler
# ─────────────────────────────────────────────
def handle_add(filepath: str):

    rag = st.session_state.rag

    if not os.path.isfile(filepath):
        return f"❌ File not found: `{filepath}`"

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    rag.add_code_file(filepath, content)
    return f"✅ Added `{filepath}` to knowledge base"


# ─────────────────────────────────────────────
# Streaming Chat Handler
# ─────────────────────────────────────────────
def stream_chat_response(prompt: str):

    generator = st.session_state.generator

    response_text = ""

    # Create assistant message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Stream tokens directly
        for chunk in generator.stream_inference(
            generator._build_prompt(
                user_msg=prompt,
                rag_context=generator.rag.query(prompt),
                project_context=generator._retrieve_project_context(prompt),
            )
        ):
            response_text += chunk
            message_placeholder.markdown(response_text + "▌")

        # Final render without cursor
        message_placeholder.markdown(response_text)

    return response_text


# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask for code or generate a project..."):

    generator = st.session_state.generator

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ─────────────────────────────
    # Command Routing
    # ─────────────────────────────
    if prompt.lower() == "/clear":
        generator.conversation_history.clear()
        st.session_state.messages = []
        st.rerun()

    elif prompt.lower().startswith("/project "):
        description = prompt[9:].strip()
        response = handle_project(description)

        with st.chat_message("assistant"):
            st.markdown(response)

    elif prompt.lower().startswith("/add "):
        filepath = prompt[5:].strip()
        response = handle_add(filepath)

        with st.chat_message("assistant"):
            st.markdown(response)

    elif prompt.lower() in ("/quit", "/exit"):
        response = "👋 Goodbye! Refresh the page to restart."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.stop()

    else:
        # 🔥 REAL STREAMING
        response = stream_chat_response(prompt)

        # Update generator memory AFTER streaming
        generator.conversation_history.append(
            {"role": "user", "content": prompt}
        )
        generator.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        generator.rag.add_conversation(prompt, response)

    # Save assistant message to session history
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )