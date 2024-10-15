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