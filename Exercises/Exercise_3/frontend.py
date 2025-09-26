import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="Sentiment Analyzer", page_icon="😊")

# Sidebar
with st.sidebar:
    st.title("😊 Sentiment Analyzer")
    st.markdown("*detect positive or negative tone*")
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

# Main Chat Section
st.subheader("🗯 Chat with Sentiment Analyzer")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_box = st.container(height=500, border=True)

# Display chat history
with chat_box:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])


user_input = st.chat_input("Enter a sentence to analyze sentiment...")
if user_input:
 
    st.session_state.messages.append({"role": "user", "text": user_input})

    
    with st.spinner("Analyzing sentiment..."):
        try:
            url = "http://127.0.0.1:8000/sentiment"  
            payload = {"text": user_input}
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            ai_reply = f"**Sentiment:** {data['label']}  \n**Confidence:** {data['score']}"
        except requests.exceptions.RequestException as e:
            ai_reply = f"⚠️ Error: {e}"

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "text": ai_reply})
    st.rerun()
