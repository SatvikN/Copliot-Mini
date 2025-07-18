import sys
from pathlib import Path

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
from backend.rag.faiss_index import FaissRAGIndex

# Supported file extensions
CODE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.go', '.cpp', '.c', '.cs', '.rb'}
DOC_EXTENSIONS = {'.md', '.txt'}

# --- Chunking Utilities ---
def chunk_markdown(text: str) -> List[str]:
    # Split by headers, then by paragraphs
    header_chunks = re.split(r'(^#+ .*$)', text, flags=re.MULTILINE)
    chunks = []
    buffer = ''
    for part in header_chunks:
        if part.strip().startswith('#'):
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = part.strip() + '\n'
        else:
            buffer += part
    if buffer.strip():
        chunks.extend([p.strip() for p in buffer.split('\n\n') if p.strip()])
    return [c for c in chunks if len(c.split()) > 10]  # Filter very short chunks

def chunk_python_code(text: str) -> List[Tuple[str, Dict]]:
    import ast
    chunks = []
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = max([getattr(n, 'end_lineno', start) for n in ast.walk(node)])
                lines = text.splitlines()[start:end]
                chunk = '\n'.join(lines)
                if len(chunk.split()) > 5:
                    chunks.append((chunk, {'type': type(node).__name__, 'name': getattr(node, 'name', None), 'start': start, 'end': end}))
    except Exception:
        pass
    return chunks

def chunk_code_lines(text: str, window: int = 20, overlap: int = 5) -> List[Tuple[str, Dict]]:
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), window - overlap):
        chunk = '\n'.join(lines[i:i+window])
        if len(chunk.split()) > 5:
            chunks.append((chunk, {'type': 'lines', 'start': i, 'end': i+window}))
    return chunks

# --- Main Indexing Script ---
def index_project(root_dirs: List[str], faiss_index_path: str = None):
    if faiss_index_path is None:
        faiss_index_path = os.path.join(os.path.dirname(__file__), 'index_data')
    
    rag_index = FaissRAGIndex()
    documents = []
    file_count = 0
    
    for root_dir in root_dirs:
        print(f"Scanning directory: {root_dir}")
        if not os.path.exists(root_dir):
            print(f"Warning: Directory {root_dir} does not exist, skipping...")
            continue
            
        for path in Path(root_dir).rglob('*'):
            if path.is_file():
                file_count += 1
                if file_count % 10 == 0:
                    print(f"Processed {file_count} files...")
                
                ext = path.suffix.lower()
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    if ext in DOC_EXTENSIONS:
                        chunks = chunk_markdown(text)
                        for chunk in chunks:
                            documents.append((chunk, {'file': str(path), 'type': 'doc'}))
                        if chunks:
                            print(f"  Added {len(chunks)} doc chunks from {path.name}")
                            
                    elif ext == '.py':
                        code_chunks = chunk_python_code(text)
                        if not code_chunks:
                            code_chunks = chunk_code_lines(text)
                        for chunk, meta in code_chunks:
                            meta.update({'file': str(path)})
                            documents.append((chunk, meta))
                        if code_chunks:
                            print(f"  Added {len(code_chunks)} code chunks from {path.name}")
                            
                    elif ext in CODE_EXTENSIONS:
                        code_chunks = chunk_code_lines(text)
                        for chunk, meta in code_chunks:
                            meta.update({'file': str(path)})
                            documents.append((chunk, meta))
                        if code_chunks:
                            print(f"  Added {len(code_chunks)} code chunks from {path.name}")
                            
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
    
    print(f"\nTotal files processed: {file_count}")
    print(f"Total chunks extracted: {len(documents)}")
    
    if documents:
        rag_index.add_documents(documents)
        rag_index.save(faiss_index_path)
        print(f"Indexed {len(documents)} chunks and saved to {faiss_index_path}")
    else:
        print("No documents found to index!")
    
    return rag_index

if __name__ == '__main__':
    # Example usage: index codebase and docs
    project_dirs = ['docs', 'backend']  # Use correct relative paths from project root
    index_project(project_dirs) 