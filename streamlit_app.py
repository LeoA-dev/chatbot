import streamlit as st
from openai import OpenAI
import openai
import time
import json
import os

# Title
st.title("💬 Analysen Chatbot with Selectable Histories")
st.write("Chat with GPT-4o-mini and continue or manage multiple conversations.")

# Secrets
openai_api_key = st.secrets.openai_api_key
ASSISTANT_ID = st.secrets.ASSISTANT_ID

client = OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key

# Constants
CHAT_DIR = "chat_histories"
os.makedirs(CHAT_DIR, exist_ok=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# --- Load Chat ---
chat_files = [f[:-5] for f in os.listdir(CHAT_DIR) if f.endswith(".json")]
selected_chat = st.selectbox("📂 Load a saved chat to continue", ["Select a chat..."] + chat_files)

if selected_chat != "Select a chat..." and selected_chat != st.session_state.current_chat:
    with open(os.path.join(CHAT_DIR, f"{selected_chat}.json"), "r") as f:
        data = json.load(f)
        st.session_state.messages = data.get("messages", [])
        st.session_state.thread_id = data.get("thread_id")
        st.session_state.current_chat = selected_chat
    st.success(f"✅ Loaded and ready to continue: {selected_chat}")
    st.rerun()

# --- Save Chat ---
with st.expander("💾 Save Current Chat"):
    chat_name = st.text_input("Enter a name for this chat history", value=st.session_state.current_chat or "")
    if st.button("Save Chat") and chat_name.strip():
        path = os.path.join(CHAT_DIR, f"{chat_name.strip()}.json")
        data = {
            "messages": st.session_state.messages,
            "thread_id": st.session_state.thread_id
        }
        with open(path, "w") as f:
            json.dump(data, f)
        st.success(f"💾 Chat saved as: {chat_name.strip()}")
        st.session_state.current_chat = chat_name.strip()

# --- Display Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Response ---
if prompt := st.chat_input("Say something..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to assistant via existing thread
    openai.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt
    )

    run = openai.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=ASSISTANT_ID
    )

    with st.spinner("Thinking..."):
        while True:
            run_status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            time.sleep(1)

    messages = openai.beta.threads.messages.list(
        thread_id=st.session_state.thread_id
    )
    reply = messages.data[0].content[0].text.value

    # Show and save assistant reply
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
