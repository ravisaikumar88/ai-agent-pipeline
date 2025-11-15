import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_qdrant import Qdrant as QdrantVectorStore # Qdrant is imported as QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
from config import PDF_PATH, COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, embeddings

# --- 1. Qdrant Client Initialization ---
def initialize_qdrant_client():
    """Initializes the Qdrant client for remote connection."""
    print("Connecting to Qdrant Cloud...")
    try:
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY
        )
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        # Use simple error logging/exit since this is a setup script
        sys.exit(1)

# --- 2. Check and Setup Collection ---
def setup_collection(client: QdrantClient, embedding_size: int):
    """Checks if the collection exists and creates it if not."""
    
    # Check if collection exists
    collections = client.get_collections().collections
    if COLLECTION_NAME in [c.name for c in collections]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        # Ensure collection is ready (optional, but good practice)
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            if collection_info.status != CollectionStatus.GREEN:
                print(f"Collection status is {collection_info.status}. Waiting might be required.")
        except Exception:
            # Handle case where collection exists but info cannot be fetched
            pass 
        return True # Collection exists and is ready to use
    
    # Create the collection
    print(f"Collection '{COLLECTION_NAME}' not found. Creating new collection...")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
        return False # Collection was just created
    except Exception as e:
        print(f"Error creating collection: {e}")
        sys.exit(1)


# --- 3. Main RAG Setup Function ---
def setup_rag_pipeline():
    """Loads, chunks, embeds, and uploads documents to Qdrant, returning the retriever."""
    if not os.path.exists(PDF_PATH):
        print(f"Error: Document not found at '{PDF_PATH}'. Please ensure the file exists.")
        sys.exit(1)

    # 1. Load Document
    print(f"Loading document from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 2. Split Document
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # 3. Initialize Qdrant Client and Setup Collection
    qdrant_client = initialize_qdrant_client()
    
    # Determine embedding size dynamically
    try:
        # NOTE: Embedding a simple query to get the expected vector dimension
        dummy_vector = embeddings.embed_query("test query")
        embedding_size = len(dummy_vector)
    except Exception as e:
        print(f"Could not determine embedding size. Error: {e}")
        sys.exit(1)
        
    was_already_populated = setup_collection(qdrant_client, embedding_size)
    
    if was_already_populated:
        # If collection exists, assume it is populated and return the retriever
        print(f"Collection '{COLLECTION_NAME}' assumed to be populated. Returning retriever.")
        
        # --- FIX APPLIED HERE: Pass the qdrant_client object instead of URL/API Key ---
        return QdrantVectorStore(
            client=qdrant_client, # Pass the initialized client object
            embeddings=embeddings,
            collection_name=COLLECTION_NAME
        ).as_retriever(search_kwargs={"k": 3})
        # --- END FIX ---

    # 4. Create and Upload Embeddings (Only if collection was newly created)
    print(f"Generating embeddings and uploading {len(docs)} documents to Qdrant...")
    # NOTE: The from_documents method correctly handles client creation internally 
    # if url/api_key are provided, but we pass the client for consistency.
    vector_store = QdrantVectorStore.from_documents(
        docs, 
        embeddings, 
        client=qdrant_client, # Pass the initialized client object
        collection_name=COLLECTION_NAME,
        force_recreate=False # Handled by setup_collection
    )
    print("Document embedding and Qdrant upload complete.")
    
    # Return the retriever instance
    return vector_store.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    print("--- Starting RAG Pipeline Setup ---")
    retriever = setup_rag_pipeline()
    print("RAG setup complete. Retriever is ready.")