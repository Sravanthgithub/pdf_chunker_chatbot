import os
import fitz
import pdfplumber
import tempfile
import pandas as pd

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