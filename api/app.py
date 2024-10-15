import os
import sys
import fitz
import pdfplumber
import openai
import faiss
import tempfile
import numpy as np
import pandas as pd
from openai import OpenAI
import base64
from PIL import Image
import io
import json
import hashlib
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY

def parse_pdf(pdf_path: str) -> tuple:
    """
    Parse a PDF file and extract text, tables, and images.
    
    Args:
    pdf_path (str): Path to the PDF file.
    
    Returns:
    tuple: A tuple containing content_items and page_image_map.
    """
    content_items = []
    page_image_map = {}
    
    with pdfplumber.open(pdf_path) as pdf, fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                content_items.append({'type': 'text', 'page': page_num + 1, 'content': text.strip()})
            
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                content_items.append({'type': 'table', 'page': page_num + 1, 'content': df.to_csv(index=False)})
            
            page_images = doc.load_page(page_num).get_images(full=True)
            page_image_map[page_num + 1] = []
            for img_index, img in enumerate(page_images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"page{page_num+1}_img{img_index}.{image_ext}"
                image_path = os.path.join(tempfile.gettempdir(), image_name)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                page_image_map[page_num + 1].append(image_path)
                content_items.append({'type': 'image', 'page': page_num + 1, 'content': image_path})
    
    return content_items, page_image_map

def chunk_content(content_items: list, max_size: int = 1000) -> list:
    """
    Chunk the content items into smaller pieces.
    
    Args:
    content_items (list): List of content items.
    max_size (int): Maximum size of each chunk.
    
    Returns:
    list: List of chunked content items.
    """
    chunks = []
    current_chunk = ''
    current_page = None
    
    for item in content_items:
        if item['type'] == 'text':
            sentences = item['content'].split('. ')
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_size:
                    current_chunk += sentence + '. '
                    current_page = item['page']
                else:
                    if current_chunk:
                        chunks.append({'type': 'text', 'content': current_chunk.strip(), 'page': current_page})
                    current_chunk = sentence + '. '
                    current_page = item['page']
            if current_chunk:
                chunks.append({'type': 'text', 'content': current_chunk.strip(), 'page': current_page})
                current_chunk = ''
        elif item['type'] in ['table', 'image']:
            chunks.append(item)
    
    return chunks

def get_embedding(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Get the embedding for a given text.
    
    Args:
    text (str): Input text.
    model (str): Model to use for embedding.
    
    Returns:
    np.ndarray: Embedding vector.
    """
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model, dimensions=1024).data[0].embedding
    return np.array(embedding).reshape(1, -1)

def encode_image(image_path: str) -> str:
    """
    Encode an image to base64.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    str: Base64 encoded image.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_image_description(image_path: str) -> str:
    """
    Get a description of an image using GPT-4 Vision.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    str: Description of the image.
    """
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def create_embeddings(chunks: list) -> tuple:
    """
    Create embeddings for the chunks.
    
    Args:
    chunks (list): List of content chunks.
    
    Returns:
    tuple: A tuple containing embeddings and chunk_texts.
    """
    embeddings = []
    chunk_texts = []
    for chunk in chunks:
        if chunk['type'] in ['text', 'table']:
            chunk_text = chunk['content']
        elif chunk['type'] == 'image':
            chunk_text = get_image_description(chunk['content'])
            chunk['description'] = chunk_text
        
        embedding = get_embedding(chunk_text)
        embeddings.append(embedding)
        chunk_texts.append(chunk)
    
    return np.array(embeddings).astype('float32'), chunk_texts

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

def query_gpt(prompt: str, model: str = "gpt-4") -> str:
    """
    Query GPT model with a prompt.
    
    Args:
    prompt (str): Input prompt.
    model (str): GPT model to use.
    
    Returns:
    str: Generated response.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return None

def answer_question(question: str, chunks: list, chunk_texts: list, index: faiss.IndexFlatL2, 
                    top_k: int = 5, approach: int = 1, page_image_map: dict = None) -> tuple:
    """
    Answer a question based on the provided context and approach.
    
    Args:
    question (str): Input question.
    chunks (list): List of content chunks.
    chunk_texts (list): List of chunk texts.
    index (faiss.IndexFlatL2): FAISS index.
    top_k (int): Number of top relevant chunks to consider.
    approach (int): Approach to use (1 or 2).
    page_image_map (dict): Mapping of pages to images.
    
    Returns:
    tuple: A tuple containing the answer and relevant image paths.
    """
    if approach == 1:
        top_k = 10
    else:
        top_k = 5
        
    question_embedding = get_embedding(question)
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [chunk_texts[i] for i in indices[0]]
    context = ''
    image_paths = []
    relevant_pages = set()

    for chunk in relevant_chunks:
        if chunk['type'] in ['text', 'table']:
            context += chunk['content'] + '\n'
        if approach == 1 and chunk['type'] == 'image':
            image_paths.append(chunk['content'])
            context += f"Image description: {chunk['description']}\n"
        elif approach == 2:
            relevant_pages.add(chunk['page'])
    
    if approach == 2:
        for page in relevant_pages:
            if page in page_image_map:
                image_paths.extend(page_image_map[page])
                context += f"Images from page {page} are relevant.\n"
    
    if image_paths and approach == 1:
        prompt = f"Answer the following question based on the image and context provided. If the answer is not in the image, use the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = query_gpt(prompt, model="gpt-4-vision-preview")
    else:
        prompt = f"Use the following information to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = query_gpt(prompt)
    
    return answer, image_paths

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
    cache_dir = os.path.join(os.path.dirname(pdf_path), f"{pdf_hash}_cache")
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

def main():
    pdf_path = input("Enter the path to your PDF file: ")
    if not os.path.isfile(pdf_path):
        print(f"File {pdf_path} does not exist.")
        sys.exit(1)
    
    print("Choose an approach:")
    print("1. Image embedding with GPT-4 Vision")
    print("2. Page-based image mapping")
    approach = int(input("Enter 1 or 2: "))
    
    if approach not in [1, 2]:
        print("Invalid approach selected. Exiting.")
        return
    
    print("Loading or creating index...")
    index, chunks, chunk_texts, page_image_map = load_or_create_index(pdf_path, approach)
    
    print("Ready to answer questions. Type 'exit' to quit.")
    
    while True:
        question = input("Please enter your question: ")
        if question.lower() == 'exit':
            break
        print("Generating answer...")
        answer, image_paths = answer_question(question, chunks, chunk_texts, index, top_k=10, approach=approach, page_image_map=page_image_map)
        print("Answer:", answer)
        if image_paths:
            print("\nRelevant images:", ', '.join(image_paths))
        print()

if __name__ == '__main__':
    main()