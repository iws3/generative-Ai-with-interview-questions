# streamlit_app.py
import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Summarizer LLM", page_icon="📝")

# Sidebar
with st.sidebar:
    st.title("📝 Summarizer LLM")
    st.markdown("*Summarize long text with BART Large CNN*")
    st.markdown("___")
    
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Summaries")
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
    max_length = st.slider("Max Length", min_value=50, max_value=500, value=150, step=10)
    min_length = st.slider("Min Length", min_value=10, max_value=100, value=30, step=5)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, step=1)
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    st.markdown("___")

# Main Section
st.subheader("🗯 Summarize Your Text")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_box = st.container(height=500, border=True)

# Display history
with chat_box:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

# Input area
user_input = st.chat_input("Paste your long text here...")
if user_input:
    
    st.session_state.messages.append({"role": "user", "text": user_input})

    with st.spinner("AI is summarizing your text..."):
        try:
            url = "http://127.0.0.1:8000/summarize" 
            payload = {
                "text": user_input,
                "max_length": max_length,
                "min_length": min_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            ai_reply = data.get("summary_text", "⚠️ No summary returned.")
        except requests.exceptions.RequestException as e:
            ai_reply = f"⚠️ Error: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "text": ai_reply})
    st.rerun()
