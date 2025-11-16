import streamlit as st
from agent_core import app

st.set_page_config(
    page_title="🤖 AI Agent Pipeline",
    page_icon="🔗",
    layout="wide",
)

st.title("AI Agent Pipeline 🔗🤖")
st.caption("Ask me about the weather or query the knowledge base (the dummy PDF).")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's the weather in Berlin or what is LangGraph?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (The agent is routing and fetching data)"):
            try:
                inputs = {"query": prompt}
                response = app.invoke(inputs)
                
                final_answer = response.get("final_response", "Error: No final response generated.")

            except Exception as e:
                final_answer = f"An error occurred while processing your request: {e}"
                print(f"Streamlit App Error: {e}")

        st.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

