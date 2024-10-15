# PDF Chunker Chatbot

PDF Chunker Chatbot is a Python application that allows users to extract information from PDF documents and ask questions about their content. The application uses advanced natural language processing techniques to understand and respond to user queries based on the content of the uploaded PDF. Main thing, this can be used if your pdf even has tables and images.

## Features

- PDF parsing: Extract text, tables, and images from PDF documents
- Content chunking: Break down large documents into manageable pieces
- Embeddings generation: Create vector representations of text and image content
- Indexing: Efficiently store and retrieve relevant information
- Question answering: Provide accurate answers to user queries based on the PDF content
- Support for two approaches:
  1. Image embedding with GPT-4 Vision
  2. Page-based image mapping

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Sravanthgithub/pdf-chunker-chatbot.git
   cd pdf-chunker-chatbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Enter the path to your PDF file when prompted.

3. Choose an approach:
   - 1 for Image embedding with GPT-4 Vision
   - 2 for Page-based image mapping

4. Once the index is created or loaded, you can start asking questions about the PDF content.

5. Type 'exit' to quit the application.

## Project Structure

```
pdf_chunker_chatbot/
│
├── app/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── indexer.py
│   └── question_answering.py
│
├── main.py
├── .env
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- OpenAI for providing the GPT and embedding models
- FAISS for efficient similarity search
- PyMuPDF and pdfplumber for PDF parsing capabilities

## Disclaimer

This tool is for educational and research purposes only. Ensure you have the right to use and analyze the PDFs you process with this tool.