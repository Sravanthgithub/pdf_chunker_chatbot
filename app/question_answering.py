from openai import OpenAI
import faiss
import numpy as np
from app.embeddings import get_embedding

client = OpenAI()

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
        top_k = 10
        
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
            for page_no, page_image_paths in page_image_map.items():
                for image_path in page_image_paths:
                    image_paths.append(image_path)
                context += f"Image on page {page_no}\n"
    
    if image_paths and approach == 1:
        prompt = f"Answer the following question based on the image and context provided. If the answer is not in the image, use the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = query_gpt(prompt, model="gpt-4-vision-preview")
    else:
        prompt = f"Use the following information to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = query_gpt(prompt)
    
    image_paths = set(image_paths)
    
    return answer, image_paths