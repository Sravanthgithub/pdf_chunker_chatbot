import os
import json
import faiss
import hashlib
from app.pdf_parser import parse_pdf
from app.chunker import chunk_content
from app.embeddings import create_embeddings
import numpy as np

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS index from embeddings.
    
    Args:
    embeddings (np.ndarray): Array of embeddings.
    
    Returns:
    faiss.IndexFlatL2: FAISS index.
    """
    index = faiss.IndexFlatL2(1024)
    for emb in embeddings:
        index.add(emb)
    return index

def get_pdf_hash(pdf_path: str) -> str:
    """
    Generate a hash for the PDF file.
    
    Args:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    str: Hash of the PDF file.
    """
    with open(pdf_path, "rb") as f:
        pdf_hash = hashlib.md5(f.read()).hexdigest()
    return pdf_hash

def load_or_create_index(pdf_path: str, approach: int) -> tuple:
    """
    Load existing index or create a new one for the PDF.
    
    Args:
    pdf_path (str): Path to the PDF file.
    approach (int): Approach to use (1 or 2).
    
    Returns:
    tuple: A tuple containing index, chunks, chunk_texts, and page_image_map.
    """
    pdf_hash = get_pdf_hash(pdf_path)
    store_dir = "/mnt/private/personal/pdf_chunker_chatbot/cache"
    cache_dir = os.path.join(store_dir, f"{pdf_hash}_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    index_path = os.path.join(cache_dir, f"index_approach_{approach}.faiss")
    data_path = os.path.join(cache_dir, f"data_approach_{approach}.json")
    
    if os.path.exists(index_path) and os.path.exists(data_path):
        index = faiss.read_index(index_path)
        with open(data_path, 'r') as f:
            data = json.load(f)
        chunks = data['chunks']
        chunk_texts = data['chunk_texts']
        page_image_map = data['page_image_map']
    else:
        content_items, page_image_map = parse_pdf(pdf_path)
        chunks = chunk_content(content_items)
        embeddings, chunk_texts = create_embeddings(chunks)
        index = build_faiss_index(embeddings)
        
        faiss.write_index(index, index_path)
        with open(data_path, 'w') as f:
            json.dump({
                'chunks': chunks,
                'chunk_texts': chunk_texts,
                'page_image_map': page_image_map
            }, f)
    
    return index, chunks, chunk_texts, page_image_map