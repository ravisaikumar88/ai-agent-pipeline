AI Agent Pipeline (LangChain, LangGraph, Qdrant & Streamlit)

This project implements a simple, robust AI agentic pipeline as per the AI Engineer assignment. The agent is capable of making decisions to either fetch real-time weather data or answer questions from a private document (RAG).

The entire application is built using a modular structure, fully tested, and observable via LangSmith.

Core Functionalities

RAG (Retrieval-Augmented Generation): The agent can answer questions about the contents of the dummy_doc.pdf file. This uses a HuggingFace embedding model and a Qdrant vector database.

Tool Use (API): The agent can fetch real-time weather for any city in the world using the OpenWeatherMap API.

Dynamic Routing (LangGraph): A central router node analyzes the user's query and decides which tool to use (RAG, Weather, or a rejection for out-of-scope questions).

Observability (LangSmith): All agent runs, decisions, and LLM calls are traced and logged in LangSmith for evaluation and debugging.

Project Structure

The project is broken down into modular, single-responsibility files for clean code:

.env: (Must be created) Holds all API keys.

.env.example: A template file for the required API keys.

.gitignore: Ensures secret keys and environment files are not uploaded to GitHub.

requirements.txt: All Python dependencies.

config.py: Loads environment variables and initializes core components (LLM, Embeddings).

dummy_doc.pdf: The knowledge base for the RAG system.

rag_setup.py: A one-time setup script to load, chunk, embed, and store the PDF in the Qdrant vector store.

tools.py: Contains the fetch_weather_data function.

agent_core.py: The "brain" of the operation. Defines the LangGraph agent state, nodes, and graph compilation.

test_logic.py: pytest unit tests for the router and tool logic (uses mocking).

app.py: The final Streamlit web application.

Setup & Installation

Follow these steps to run the application locally.

1. Prerequisites

Python 3.10+

API Keys for:

Groq (for the LLM)

Qdrant Cloud (for the vector database)

OpenWeatherMap (for the weather tool)

LangSmith (for observability)

2. Create your Environment

First, create a .env file in the root of the project by copying the .env.example template and pasting in your API keys.

# .env file

# Groq API
GROQ_API_KEY=gsk_...

# LangSmith
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AI_Agent_Pipeline

# Vector DB (Qdrant)
QDRANT_API_KEY=...
QDRANT_URL=https://...

# Weather API
OPENWEATHER_API_KEY=...


3. Install Dependencies

Create a virtual environment and install the required packages.

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt


4. Populate the Vector Database

Run the one-time setup script to process the PDF and populate your Qdrant database.

python rag_setup.py


5. Run Unit Tests (Optional)

Verify that all the logic is working correctly using pytest.

pytest


You should see 5 passed.

6. Run the Streamlit Application

You are now ready to launch the chat UI.

streamlit run app.py


This will open the application in your web browser.

LangSmith Evaluation

All runs are automatically traced to LangSmith by setting the environment variables. The agent's decision-making (router node), tool calls, and final synthesis (synthesis node) can be clearly inspected. This allows for debugging complex runs and evaluating the quality of the agent's responses.

Troubleshooting

Error: ConnectTimeoutError or Connection timed out for OpenWeatherMap

This is a local network issue, not a code bug. The requests library is being blocked by a local firewall.

Solution:

Open "Firewall & network protection" on Windows.

Click "Allow an app through firewall".

Click "Change settings" (admin required).

Click "Allow another app..."

Browse to the python.exe file inside your virtual environment:
[PATH_TO_YOUR_PROJECT]\.venv\Scripts\python.exe

Click "Add" and check both "Private" and "Public" boxes.

Relaunch the app.