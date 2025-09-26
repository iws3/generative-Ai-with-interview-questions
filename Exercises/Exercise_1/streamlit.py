# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Hello LLM", page_icon="🤖")

# Sidebar
with st.sidebar:
    st.title("🤖 Hello LLM")
    st.markdown("*play with text generation*")
    st.markdown("___")
    
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Messages")
        if "messages" in st.session_state:
            st.markdown(f"# {len(st.session_state.messages) // 2}")
        else:
            st.markdown("# 0")
    with col2:
        st.header("Total")
        if "messages" in st.session_state:
            st.markdown(f"# {len(st.session_state.messages)}")
        else:
            st.markdown("# 0")
    st.markdown("___")

    st.header("⚙ Controls")
    max_length = st.slider("Max Length", min_value=10, max_value=200, value=50, step=5)
    st.markdown("___")

# Main Chat Section
st.subheader("🗯 Chat with Hello LLM")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_box = st.container(height=500, border=True)

# Display chat history
with chat_box:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

# Input area
user_input = st.chat_input("Type your prompt here...")
if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "text": user_input})

    # Send to FastAPI
    with st.spinner("AI is generating response..."):
        try:
            url = "http://127.0.0.1:8000/hello-llm"  # FastAPI endpoint
            payload = {"text": user_input, "max_length": max_length}
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            ai_reply = data["generated_text"]
        except requests.exceptions.RequestException as e:
            ai_reply = f"⚠️ Error: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "text": ai_reply})
    st.rerun()
