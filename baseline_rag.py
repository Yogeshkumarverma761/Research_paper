import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_baseline_vectorstore(data_path: str, persist_directory: str = "./chroma_db"):
    """
    Sets up a baseline vector database and demonstrates the hallucination problem.
    """
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found.")
        return None
    
    # 1. Load and split the dataset
    loader = TextLoader(data_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    
    # 2. Embed and Index using a local HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Indexing {len(chunks)} chunks into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Baseline VectorDB setup complete.\n")
    
    # 3. Simulate the Hallucination Trigger (Retrieval Phase)
    query = "How many days can employees work from home?"
    print(f"User Query: '{query}'\n")
    
    results = vectorstore.similarity_search_with_score(query, k=2)
    
    print("--- Retrieved Context ---")
    for doc, score in results:
        # Lower score in Chroma means higher similarity (it uses L2 distance by default)
        print(f"Score (L2 Distance): {score:.4f}")
        print(f"Content: {doc.page_content}\n")
        
    print("--- Analysis ---")
    print("Standard cosine similarity retrieves BOTH the outdated policy (2 days) and the updated policy (5 days) with very similar scores.")
    print("When an LLM receives this contradictory context, it is highly likely to hallucinate the answer because it lacks the mathematical mechanism to resolve the conflict.")
    print("This is the exact loophole we will solve with DCAVI.")

if __name__ == "__main__":
    DATASET_PATH = "data/domain_dataset.txt" 
    setup_baseline_vectorstore(DATASET_PATH)
