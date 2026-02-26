"""
Offline AI Code Generator — With Authentication
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
from auth import AuthManager


# ─────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Offline AI Code Generator", page_icon="🤖")
st.title("🤖 Offline AI Code Generator")

auth = AuthManager()


# ─────────────────────────────────────────────
# Authentication State
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None


# ─────────────────────────────────────────────
# LOGIN / REGISTER UI
# ─────────────────────────────────────────────
def login_ui():

    st.subheader("🔐 Login or Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # LOGIN
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if auth.login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # REGISTER
    with tab2:
        new_user = st.text_input("Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")

        if st.button("Register"):
            if auth.register(new_user, new_pass):
                st.success("User registered! Please login.")
            else:
                st.error("Username already exists")


# ─────────────────────────────────────────────
# STOP APP IF NOT AUTHENTICATED
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    login_ui()
    st.stop()


# ─────────────────────────────────────────────
# LOGOUT BUTTON
# ─────────────────────────────────────────────
with st.sidebar:
    st.write(f"👤 Logged in as: **{st.session_state.username}**")

    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


# ─────────────────────────────────────────────
# Load AI System (Cached)
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
# Display Chat History
# ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────
# Streaming Chat
# ─────────────────────────────────────────────
def stream_chat(prompt):

    generator = st.session_state.generator
    response_text = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()

        for chunk in generator.stream_inference(
            generator._build_prompt(
                user_msg=prompt,
                rag_context=generator.rag.query(prompt),
                project_context=generator._retrieve_project_context(prompt),
            )
        ):
            response_text += chunk
            placeholder.markdown(response_text + "▌")

        placeholder.markdown(response_text)

    return response_text


# ─────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask for code..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = stream_chat(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )