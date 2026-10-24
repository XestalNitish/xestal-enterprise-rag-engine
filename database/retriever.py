import os 
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_classic.retrievers import EnsembleRetriever # ya jahan se bhi import kiya hai

@tool
def retrieve(query: str, workspace_id: str) -> list:
    """
    Creates a Hybrid Retriever combining BM25 (Keyword Search) and Chroma (Vector/Semantic Search).
    Returns a list of dictionaries containing text and source for proper chunk normalization.
    """
    try:
        file_path = f"./workspace_docs/{workspace_id}_docs.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                documents = pickle.load(f)
        else:
            return []

        if not documents:
            return []

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # FIX: Workspace-isolated directory/collection
        vector_store = Chroma(
            persist_directory=f"./test_db_store_{workspace_id}",
            embedding_function=embeddings,
            collection_name=f"enterprise_docs_{workspace_id}"
        )
        
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = 10
        
        vector_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4, 
                "fetch_k": 20,    
                "lambda_mult": 0.7,  
                "filter": {"workspace_id": workspace_id}
            },
        ) 
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_retriever],
            weights=[0.3, 0.7]
        )
        
        final_docs = hybrid_retriever.invoke(query)
        print("Hybrid Retriever executed successfully!")
        
        # FIX: Return structured dicts so normalize_chunks can read source and text properly
        formatted_chunks = []
        for doc in final_docs:
            source = doc.metadata.get("source", f"workspace_{workspace_id}")
            formatted_chunks.append({
                "text": doc.page_content,
                "source": source
            })
            
        return formatted_chunks
        
    except Exception as e:
        print(f"[ERROR] Hybrid retriever creation failed: {str(e)}")
        return []