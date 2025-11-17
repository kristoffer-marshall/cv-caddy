"""
Text processing utilities for chunking and name extraction.
"""
import re


def split_text_into_chunks(text, chunk_size, overlap):
    """
    Splits text into overlapping chunks without a library.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def extract_name_from_text(text):
    """
    Attempts to extract a name from text. Looks for common patterns:
    - First line that looks like a name (2-4 capitalized words)
    - Patterns like "Name:", "My name is", etc.
    Returns the first likely name found, or None.
    """
    if not text:
        return None
    
    lines = text.strip().split('\n')
    
    # Check first few lines for a name pattern (typically at top of resume)
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        
        # Remove common prefixes
        line_clean = re.sub(r'^(Name|NAME|Full Name|Full Name:)\s*:?\s*', '', line, flags=re.IGNORECASE)
        line_clean = line_clean.strip()
        
        # Look for pattern: 2-4 capitalized words (likely a name)
        name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$'
        match = re.match(name_pattern, line_clean)
        if match:
            name = match.group(1)
            # Filter out common non-name words
            if not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae', 'email', 'phone', 'address'] 
                      for word in name.split()):
                return name
    
    # Check for "My name is" pattern
    name_match = re.search(r'(?:My name is|I am|I\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text, re.IGNORECASE)
    if name_match:
        return name_match.group(1)
    
    # Check for "Name:" pattern anywhere
    name_match = re.search(r'Name\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text, re.IGNORECASE)
    if name_match:
        return name_match.group(1)
    
    return None

