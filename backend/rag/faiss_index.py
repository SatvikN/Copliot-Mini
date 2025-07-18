import os
import json
from typing import List, Tuple, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissRAGIndex:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', embedding_dim: int = 384):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks: List[str] = []
        self.chunk_metadata: List[dict] = []

    def add_documents(self, documents: List[Tuple[str, dict]]):
        if not documents:
            print("Warning: No documents provided to index")
            return
        
        texts = [doc[0] for doc in documents]
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        self.text_chunks.extend(texts)
        self.chunk_metadata.extend([doc[1] for doc in documents])
        print(f"Successfully indexed {len(texts)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.text_chunks):
                results.append((self.text_chunks[idx], self.chunk_metadata[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(self.text_chunks, f)
        with open(os.path.join(path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f)

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'chunks.json'), 'r', encoding='utf-8') as f:
            self.text_chunks = json.load(f)
        with open(os.path.join(path, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.chunk_metadata = json.load(f) 