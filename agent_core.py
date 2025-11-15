import os
import json
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from config import llm, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, embeddings
from rag_setup import setup_rag_pipeline
from tools import fetch_weather_data  # External tool function

# --- 1. Define the LangGraph State (MODIFIED) ---
class AgentState(TypedDict):
    """
    Represents the state of the graph, tracking query, context, and decisions.
    """
    query: str
    context: str  # This will hold either RAG chunks or Weather data
    final_response: str
    # MODIFIED: Added 'END_TURN' as a possible next step
    next_node: Literal["RAG_LOOKUP", "WEATHER_API", "END_TURN"] | None
    city_name: str | None # City name if extracted


# --- 2. Define Structured Output for Router (MODIFIED) ---
class RouterDecision(BaseModel):
    """Structured output for the router to decide the next action."""
    # MODIFIED: Added 'END_TURN'
    next_action: Literal["RAG_LOOKUP", "WEATHER_API", "END_TURN"] = Field(
        description="The determined action to take. Must be one of 'RAG_LOOKUP', 'WEATHER_API', or 'END_TURN'.",
    )
    city_name: str = Field(
        default="",
        description="The specific city name extracted from the query, only if next_action is 'WEATHER_API'. Example: 'London, UK'",
    )


# --- 3. Initialize Core Components (Retriever) ---
print("Initializing Qdrant Retriever (assuming rag_setup.py was run successfully)...")
try:
    retriever = setup_rag_pipeline()
    print("Retriever instance ready.")
except Exception as e:
    print(f"Error during RAG setup: {e}. Cannot proceed without a retriever.")
    retriever = None

# --- 4. Initialize LLM Chains ---

# Router Chain (MODIFIED PROMPT)
ROUTER_PROMPT = """
You are an intelligent routing agent. Your job is to analyze the user's 'query' and decide which of the three actions to take:
1.  'RAG_LOOKUP': If the query is about the provided PDF document (e.g., "what is langgraph", "summarize the doc").
2.  'WEATHER_API': If the query is about real-time weather (e.g., "what is the temperature in Tokyo", "forecast for Paris").
3.  'END_TURN': If the query is out of scope (e.g., general knowledge, movie release dates, history, asking for a poem).

You MUST output a JSON object conforming to the provided schema.

If the query is about weather, set 'next_action' to 'WEATHER_API' and extract the city name.
If the query is about the document, set 'next_action' to 'RAG_LOOKUP'.
If the query is out of scope, set 'next_action' to 'END_TURN'.
"""
router_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            ("user", "Query: {query}"),
        ]
    )
    | llm.with_structured_output(RouterDecision)
)

# Response Synthesis Chain (Unchanged)
SYNTHESIS_PROMPT = """
You are a helpful AI assistant. Your job is to synthesize a final, user-friendly answer based on a 'query' and the 'context' provided.

The 'context' contains the information (either from a document or an API) needed to answer the query.
Answer the user's query directly using only the provided context.
If the context is an error message, inform the user about the error.

Query: {query}

Context:
{context}
"""
synthesis_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIS_PROMPT),
            ("user", "Query: {query}\n\nContext:\n{context}"),
        ]
    )
    | llm
    | StrOutputParser()
)


# --- 5. Define LangGraph Nodes (NEW NODE ADDED) ---

def router_node(state: AgentState) -> dict:
    """
    Determines the next action (RAG, Weather, or End) and updates the state.
    """
    print("---NODE: Router---")
    query = state["query"]
    decision = router_chain.invoke({"query": query})
    
    # Using .model_dump() instead of .dict() to fix Pydantic V2 deprecation warning
    print(f"Router decision: {decision.model_dump()}")
    
    return {
        "next_node": decision.next_action,
        "city_name": decision.city_name
    }

def rag_node(state: AgentState) -> dict:
# ... (existing code, no changes) ...
    print("---NODE: RAG Lookup---")
    query = state["query"]
    
    if retriever:
        documents = retriever.invoke(query)
        # Format the context for the LLM
        context = "\n\n---\n\n".join(
            [f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
        )
    else:
        context = "Error: RAG Retriever is not initialized. Cannot answer document-related questions."
        
    return {"context": context}

def weather_node(state: AgentState) -> dict:
# ... (existing code, no changes) ...
    print("---NODE: Weather API Call---")
    city = state.get("city_name")
    
    if not city:
        return {"context": "Error: City name was not extracted by the router."}

    # Call the tool function from tools.py
    weather_data = fetch_weather_data(city)
    return {"context": weather_data}

def synthesis_node(state: AgentState) -> dict:
# ... (existing code, no changes) ...
    print("---NODE: Synthesize Response---")
    query = state["query"]
    context = state["context"]
    
    final_response = synthesis_chain.invoke({"query": query, "context": context})
    return {"final_response": final_response}

# --- NEW NODE ---
def rejection_node(state: AgentState) -> dict:
    """
    Generates a polite rejection response for out-of-scope queries.
    """
    print("---NODE: Rejection---")
    return {
        "final_response": "I'm sorry, I am a specialized AI assistant. I can only provide real-time weather information or answer questions based on the loaded document."
    }

# --- 6. Define Conditional Edges (MODIFIED) ---

def should_continue(state: AgentState) -> Literal["RAG_LOOKUP", "WEATHER_API", "END_TURN"]:
    """
    The conditional edge that routes to the correct node or ends the turn.
    """
    print("---Determining Next Step---")
    # This value was set by the router_node
    return state["next_node"]


# --- 7. Compile the LangGraph (MODIFIED) ---

# Initialize the graph
workflow = StateGraph(AgentState)

# Add the nodes (Added REJECT node)
workflow.add_node("router", router_node)
workflow.add_node("RAG_LOOKUP", rag_node)
workflow.add_node("WEATHER_API", weather_node)
workflow.add_node("SYNTHESIZE", synthesis_node)
workflow.add_node("REJECT", rejection_node) # NEW

# Set the entry point
workflow.set_entry_point("router")

# Add conditional edges from the router (MODIFIED)
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "RAG_LOOKUP": "RAG_LOOKUP",
        "WEATHER_API": "WEATHER_API",
        "END_TURN": "REJECT" # NEW PATH
    }
)

# Add normal edges from the tool nodes to the synthesis node
workflow.add_edge("RAG_LOOKUP", "SYNTHESIZE")
workflow.add_edge("WEATHER_API", "SYNTHESIZE")

# The synthesis and rejection nodes are the final steps
workflow.add_edge("SYNTHESIZE", END)
workflow.add_edge("REJECT", END) # NEW

# Compile the agent
app = workflow.compile()

# --- 8. Test the Compiled Agent (MODIFIED) ---

if __name__ == "__main__":
    if not retriever:
        print("Cannot run tests: Retriever is not initialized.")
    else:
        print("\n--- Testing Compiled Agent ---")

        # Test Case 1: Weather Query
        print("\n--- Test Case 1: Weather Query ---")
        inputs_weather = {"query": "What is the weather in Berlin, Germany?"}
        for event in app.stream(inputs_weather):
            for key, value in event.items():
                print(f"Node: {key}, Output: {value}\n")

        # Test Case 2: RAG Query
        print("\n--- Test Case 2: RAG Query ---")
        inputs_rag = {"query": "What is an agentic workflow according to the document?"}
        final_rag_response = app.invoke(inputs_rag)
        print("--- FINAL RAG RESPONSE ---")
        print(final_rag_response["final_response"])

        # Test Case 3: Out-of-Scope Query (NEW TEST)
        print("\n--- Test Case 3: Out-of-Scope Query ---")
        inputs_scope = {"query": "When is Pushpa 3 releasing?"}
        final_scope_response = app.invoke(inputs_scope)
        print("--- FINAL OUT-OF-SCOPE RESPONSE ---")
        print(final_scope_response["final_response"])