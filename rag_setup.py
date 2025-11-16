import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_qdrant import Qdrant as QdrantVectorStore # Qdrant is imported as QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
from config import PDF_PATH, COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, embeddings

def initialize_qdrant_client():
    print("Connecting to Qdrant Cloud...")
    try:
        client = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY
        )
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        sys.exit(1)

def setup_collection(client: QdrantClient, embedding_size: int):
    
    # Check if collection exists
    collections = client.get_collections().collections
    if COLLECTION_NAME in [c.name for c in collections]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            if collection_info.status != CollectionStatus.GREEN:
                print(f"Collection status is {collection_info.status}. Waiting might be required.")
        except Exception:
            pass 
        return True
    
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


def setup_rag_pipeline():
    if not os.path.exists(PDF_PATH):
        print(f"Error: Document not found at '{PDF_PATH}'. Please ensure the file exists.")
        sys.exit(1)

    print(f"Loading document from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    qdrant_client = initialize_qdrant_client()
    
    try:
        dummy_vector = embeddings.embed_query("test query")
        embedding_size = len(dummy_vector)
    except Exception as e:
        print(f"Could not determine embedding size. Error: {e}")
        sys.exit(1)
        
    was_already_populated = setup_collection(qdrant_client, embedding_size)
    
    if was_already_populated:
        print(f"Collection '{COLLECTION_NAME}' assumed to be populated. Returning retriever.")
        
        return QdrantVectorStore(
            client=qdrant_client,
            embeddings=embeddings,
            collection_name=COLLECTION_NAME
        ).as_retriever(search_kwargs={"k": 3})

    print(f"Generating embeddings and uploading {len(docs)} documents to Qdrant...")
    vector_store = QdrantVectorStore.from_documents(
        docs, 
        embeddings, 
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        force_recreate=False # Handled by setup_collection
    )
    print("Document embedding and Qdrant upload complete.")
    
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    print("--- Starting RAG Pipeline Setup ---")
    retriever = setup_rag_pipeline()
    print("RAG setup complete. Retriever is ready.")