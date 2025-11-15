import streamlit as st
from agent_core import app  # Import the compiled LangGraph agent

# --- 1. Streamlit Page Setup ---
st.set_page_config(
    page_title="🤖 AI Agent Pipeline",
    page_icon="🔗",
    layout="wide",
)

st.title("AI Agent Pipeline 🔗🤖")
st.caption("Ask me about the weather or query the knowledge base (the dummy PDF).")

# --- 2. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle User Input ---
if prompt := st.chat_input("What's the weather in Berlin or what is LangGraph?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Use a spinner while the agent is processing
        with st.spinner("Thinking... (The agent is routing and fetching data)"):
            try:
                # --- 5. Call the LangGraph Agent ---
                # We invoke the 'app' with the user's query.
                # The state ('query') is passed as the input dictionary.
                inputs = {"query": prompt}
                response = app.invoke(inputs)
                
                # The final answer is in the 'final_response' key of the state
                final_answer = response.get("final_response", "Error: No final response generated.")

            except Exception as e:
                final_answer = f"An error occurred while processing your request: {e}"
                print(f"Streamlit App Error: {e}") # Log error to console

        st.markdown(final_answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

