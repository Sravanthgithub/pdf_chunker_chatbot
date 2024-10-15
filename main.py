import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from dotenv import load_dotenv
from openai import OpenAI
from app.pdf_parser import parse_pdf
from app.chunker import chunk_content
from app.embeddings import create_embeddings
from app.indexer import build_faiss_index, load_or_create_index
from app.question_answering import answer_question

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def main():
    pdf_path = input("Enter the path to your PDF file: ")
    if not os.path.isfile(pdf_path):
        print(f"File {pdf_path} does not exist.")
        return

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