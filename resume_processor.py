"""
Resume processing: reading, chunking, embedding, and data persistence.
"""
import os
import json
import fitz  # PyMuPDF
import numpy as np
import ollama
from text_processing import split_text_into_chunks, extract_name_from_text


def process_and_embed_resume(resume_pdf_path, personal_info_txt_path, personal_info_md_path, 
                              data_dir, embedding_model, chunk_size, chunk_overlap):
    """
    Reads, chunks, and embeds the resume and personal info file.
    Saves chunks and embeddings to disk.
    Returns text_chunks, all_embeddings, personal_info_chunk_indices, and extracted_name.
    """
    chunks_file = os.path.join(data_dir, "chunks.json")
    embeddings_file = os.path.join(data_dir, "embeddings.npy")
    metadata_file = os.path.join(data_dir, "chunk_metadata.json")
    
    print("Checking for context files...")
    resume_text = ""
    personal_info_text = ""
    
    # 1. Read PDF
    if os.path.exists(resume_pdf_path):
        try:
            doc = fitz.open(resume_pdf_path)
            for page in doc:
                resume_text += page.get_text() + "\n" # Add newline separator
            
            print(f"Successfully read {len(doc)} pages from PDF '{resume_pdf_path}'.")
            doc.close()
        except Exception as e:
            print(f"Error reading PDF '{resume_pdf_path}': {e}")
            # We can continue if the other file exists
    else:
        print(f"Warning: Resume file not found at '{resume_pdf_path}'.")

    # 2. Read personal info (MD or TXT) separately
    personal_file_found = False
    # Prioritize Markdown file
    if os.path.exists(personal_info_md_path):
        try:
            with open(personal_info_md_path, "r") as f:
                personal_info_text = f.read()
            print(f"Successfully read markdown file '{personal_info_md_path}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading markdown file '{personal_info_md_path}': {e}")
    
    # If no MD file, check for TXT file as fallback
    if not personal_file_found and os.path.exists(personal_info_txt_path):
        try:
            with open(personal_info_txt_path, "r") as f:
                personal_info_text = f.read()
            print(f"Successfully read text file '{personal_info_txt_path}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading text file '{personal_info_txt_path}': {e}")

    if not personal_file_found:
        print(f"Info: Personal info file not found at '{personal_info_md_path}' or '{personal_info_txt_path}'. This is optional.")

    # 3. Check if we have any text at all
    full_text = resume_text + personal_info_text
    if not full_text.strip():
        print("Error: No text found from resume or personal info file.")
        print(f"Please make sure '{resume_pdf_path}' or '{personal_info_md_path}' or '{personal_info_txt_path}' exists.")
        return None, None, None, None
        
    print(f"Processing combined text...")
    
    # 4. Split into chunks, tracking which chunks come from personal_info
    # First, chunk the resume text
    resume_chunks = split_text_into_chunks(resume_text, chunk_size, chunk_overlap) if resume_text.strip() else []
    resume_chunk_count = len(resume_chunks)
    
    # Then, chunk the personal info text
    personal_info_chunks = split_text_into_chunks(personal_info_text, chunk_size, chunk_overlap) if personal_info_text.strip() else []
    
    # Combine chunks
    text_chunks = resume_chunks + personal_info_chunks
    
    # Track which chunk indices come from personal_info
    personal_info_chunk_indices = list(range(resume_chunk_count, len(text_chunks))) if personal_info_chunks else []
    
    # Extract name from resume (first chunk) or personal_info
    extracted_name = None
    if resume_chunks:
        extracted_name = extract_name_from_text(resume_chunks[0])
    if not extracted_name and personal_info_text:
        extracted_name = extract_name_from_text(personal_info_text)
    
    print(f"Split text into {len(text_chunks)} chunks ({len(resume_chunks)} from resume, {len(personal_info_chunks)} from personal info).")
    if extracted_name:
        print(f"Extracted name: {extracted_name}")

    # 5. Create embeddings
    print(f"Creating embeddings using '{embedding_model}'. This will take a moment...")
    all_embeddings = []
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    try:
        for i, chunk in enumerate(text_chunks):
            # We call the ollama API directly
            response = ollama.embeddings(model=embedding_model, prompt=chunk)
            all_embeddings.append(response["embedding"])
            
            if (i + 1) % 10 == 0 or (i + 1) == len(text_chunks):
                print(f"  ... processed chunk {i + 1}/{len(text_chunks)}")
        
        # 6. Save to disk
        # Save chunks as JSON
        with open(chunks_file, "w") as f:
            json.dump(text_chunks, f)
            
        # Save embeddings as numpy array
        np_embeddings = np.array(all_embeddings)
        np.save(embeddings_file, np_embeddings)
        
        # Save metadata (which chunks are from personal_info, and extracted name)
        metadata = {
            'personal_info_chunk_indices': personal_info_chunk_indices,
            'extracted_name': extracted_name
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
        
        print("Successfully created and saved chunks, embeddings, and metadata.")
        return text_chunks, np_embeddings, personal_info_chunk_indices, extracted_name

    except Exception as e:
        print(f"Error creating embeddings: {e}")
        print("Is Ollama running? Try 'ollama serve' in another terminal.")
        return None, None, None, None


def load_data_from_disk(data_dir):
    """
    Loads processed chunks and embeddings from the data directory.
    Returns text_chunks, all_embeddings, personal_info_chunk_indices, and extracted_name.
    """
    chunks_file = os.path.join(data_dir, "chunks.json")
    embeddings_file = os.path.join(data_dir, "embeddings.npy")
    metadata_file = os.path.join(data_dir, "chunk_metadata.json")
    
    print(f"Loading existing data from '{data_dir}' directory...")
    if not os.path.exists(chunks_file) or not os.path.exists(embeddings_file):
        return None, None, None, None

    try:
        with open(chunks_file, "r") as f:
            text_chunks = json.load(f)
        
        all_embeddings = np.load(embeddings_file)
        
        # Load metadata if it exists (for backward compatibility)
        personal_info_chunk_indices = []
        extracted_name = None
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    personal_info_chunk_indices = metadata.get('personal_info_chunk_indices', [])
                    extracted_name = metadata.get('extracted_name', None)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}. Personal info chunks may not be prioritized.")
        
        if extracted_name:
            print(f"Loaded extracted name: {extracted_name}")
        print("Data loaded successfully.")
        return text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name
    except Exception as e:
        print(f"Error loading data: {e}. Re-processing resume.")
        return None, None, None, None

