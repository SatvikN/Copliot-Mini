import os
from typing import List, Tuple, Dict, Optional
from backend.rag.faiss_index import FaissRAGIndex

class RAGRetriever:
    def __init__(self, index_path: Optional[str] = None):
        self.rag_index = FaissRAGIndex()
        # Try to load from default path if no specific path provided
        if index_path is None:
            index_path = os.path.join(os.path.dirname(__file__), 'index_data')
        
        if os.path.exists(index_path):
            try:
                self.rag_index.load(index_path)
                print(f"✅ Loaded RAG index from {index_path}")
            except Exception as e:
                print(f"⚠️ Failed to load RAG index from {index_path}: {e}")
        else:
            print(f"⚠️ No RAG index found at {index_path}. Run the indexer first.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for the query.
        Returns a list of dicts with 'text', 'metadata', and 'score'.
        """
        results = self.rag_index.search(query, top_k=top_k)
        return [
            {'text': text, 'metadata': meta, 'score': score}
            for text, meta, score in results
        ]

    def is_loaded(self) -> bool:
        """Check if the index is loaded and ready."""
        return len(self.rag_index.text_chunks) > 0

# Example usage
if __name__ == '__main__':
    retriever = RAGRetriever()
    if retriever.is_loaded():
        query = "How do I start the FastAPI server?"
        results = retriever.retrieve(query)
        for r in results:
            print(f"Score: {r['score']:.2f}\nFile: {r['metadata'].get('file')}\nText: {r['text'][:200]}\n---\n")
    else:
        print("RAG index not loaded. Run the indexer first.") 