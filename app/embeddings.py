import numpy as np
from openai import OpenAI
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

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