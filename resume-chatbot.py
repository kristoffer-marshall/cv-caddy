import ollama
import fitz  # PyMuPDF
import numpy as np
import os
import json
import time
import argparse
import configparser

# --- CONFIGURATION ---
RESUME_PDF_PATH = "YOUR_RESUME.pdf"
PERSONAL_INFO_TXT_PATH = "personal_info.txt" # <-- Renamed for clarity
PERSONAL_INFO_MD_PATH = "personal_info.md"   # <-- New MD path
CONFIG_FILE = "config.ini"
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"

# We will store our processed data here
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

# --- Default prompt (if config.ini is missing) ---
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional vulnerability management engineer who is talking to a recruiter or hiring manager. "
    "Do not make up information. If the answer is not in the context, "
    "politely state that you are unsure, but can get back to them later. "
    "Use the knowledge from your resume, but do no reference your resume in conversation. Act casual."
)
# ---------------------


def split_text_into_chunks(text, chunk_size=1500, overlap=200):
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

def process_and_embed_resume():
    """
    Reads, chunks, and embeds the resume and personal info file.
    Saves chunks and embeddings to disk.
    """
    print("Checking for context files...")
    full_text = ""
    
    # 1. Read PDF
    if os.path.exists(RESUME_PDF_PATH):
        try:
            doc = fitz.open(RESUME_PDF_PATH)
            for page in doc:
                full_text += page.get_text() + "\n" # Add newline separator
            
            print(f"Successfully read {len(doc)} pages from PDF '{RESUME_PDF_PATH}'.")
            doc.close()
        except Exception as e:
            print(f"Error reading PDF '{RESUME_PDF_PATH}': {e}")
            # We can continue if the other file exists
    else:
        print(f"Warning: Resume file not found at '{RESUME_PDF_PATH}'.")

    # 2. Read personal info (MD or TXT)
    personal_file_found = False
    # Prioritize Markdown file
    if os.path.exists(PERSONAL_INFO_MD_PATH):
        try:
            with open(PERSONAL_INFO_MD_PATH, "r") as f:
                full_text += f.read()
            print(f"Successfully read markdown file '{PERSONAL_INFO_MD_PATH}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading markdown file '{PERSONAL_INFO_MD_PATH}': {e}")
    
    # If no MD file, check for TXT file as fallback
    if not personal_file_found and os.path.exists(PERSONAL_INFO_TXT_PATH):
        try:
            with open(PERSONAL_INFO_TXT_PATH, "r") as f:
                full_text += f.read()
            print(f"Successfully read text file '{PERSONAL_INFO_TXT_PATH}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading text file '{PERSONAL_INFO_TXT_PATH}': {e}")

    if not personal_file_found:
        print(f"Info: Personal info file not found at '{PERSONAL_INFO_MD_PATH}' or '{PERSONAL_INFO_TXT_PATH}'. This is optional.")

    # 3. Check if we have any text at all
    if not full_text.strip():
        print("Error: No text found from resume or personal info file.")
        print(f"Please make sure '{RESUME_PDF_PATH}' or '{PERSONAL_INFO_PATH}' exists.")
        return None, None
        
    print(f"Processing combined text...")
    
    # 4. Split into chunks (was step 2)
    text_chunks = split_text_into_chunks(full_text)
    print(f"Split text into {len(text_chunks)} chunks.")

    # 5. Create embeddings (was step 3)
    print(f"Creating embeddings using '{EMBEDDING_MODEL}'. This will take a moment...")
    all_embeddings = []
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        for i, chunk in enumerate(text_chunks):
            # We call the ollama API directly
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
            all_embeddings.append(response["embedding"])
            
            if (i + 1) % 10 == 0 or (i + 1) == len(text_chunks):
                print(f"  ... processed chunk {i + 1}/{len(text_chunks)}")
        
        # 6. Save to disk (was step 4)
        # Save chunks as JSON
        with open(CHUNKS_FILE, "w") as f:
            json.dump(text_chunks, f)
            
        # Save embeddings as numpy array
        np_embeddings = np.array(all_embeddings)
        np.save(EMBEDDINGS_FILE, np_embeddings)
        
        print("Successfully created and saved chunks and embeddings.")
        return text_chunks, np_embeddings

    except Exception as e:
        print(f"Error creating embeddings: {e}")
        print("Is Ollama running? Try 'ollama serve' in another terminal.")
        return None, None

def load_data_from_disk():
    """
    Loads processed chunks and embeddings from the 'data' directory.
    """
    print("Loading existing data from 'data' directory...")
    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        return None, None

    try:
        with open(CHUNKS_FILE, "r") as f:
            text_chunks = json.load(f)
        
        all_embeddings = np.load(EMBEDDINGS_FILE)
        
        print("Data loaded successfully.")
        return text_chunks, all_embeddings
    except Exception as e:
        print(f"Error loading data: {e}. Re-processing resume.")
        return None, None

def cosine_similarity(v1, v2):
    """
    Calculates cosine similarity between two numpy vectors.
    """
    # Ensure vectors are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)

def find_relevant_chunks(query_embedding, all_embeddings, text_chunks, k=3):
    """
    Finds the top-k most similar chunks to the query.
    """
    similarities = []
    for emb in all_embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append(sim)
    
    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return the text of those chunks
    return [text_chunks[i] for i in top_k_indices]

def main():
    """
    Main function to run the chatbot.
    """
    
    # --- New: Argument Parsing ---
    parser = argparse.ArgumentParser(description="A local chatbot for your resume.")
    parser.add_argument(
        "-r", 
        "--reprocess", 
        action="store_true", 
        help="Force reprocessing of the resume and context files."
    )
    args = parser.parse_args()
    
    # --- New: Config Parsing ---
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"'{CONFIG_FILE}' not found. Creating a default config file.")
        try:
            config['Chatbot'] = {'SystemPrompt': DEFAULT_SYSTEM_PROMPT}
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            print(f"Default '{CONFIG_FILE}' created successfully.")
        except Exception as e:
            print(f"Error creating default config file: {e}")
            print("Using hard-coded default prompt.")
    else:
        try:
            config.read(CONFIG_FILE)
        except Exception as e:
            print(f"Error reading '{CONFIG_FILE}': {e}. Using default prompt.")

    # Load the prompt, falling back to the default if it's missing
    system_prompt = config.get('Chatbot', 'SystemPrompt', fallback=DEFAULT_SYSTEM_PROMPT)
    
    start_time = time.time()
    
    # 1. Load or create data (with reprocess logic)
    text_chunks, all_embeddings = None, None
    if not args.reprocess:
        text_chunks, all_embeddings = load_data_from_disk()
    
    if text_chunks is None or all_embeddings is None:
        if args.reprocess:
            print("\n--reprocess flag detected. Forcing file processing...")
        text_chunks, all_embeddings = process_and_embed_resume()
        if text_chunks is None:
            print("Exiting due to error.")
            return

    end_time = time.time()
    print(f"\n--- Ready to chat! (Setup took {end_time - start_time:.2f}s) ---")
    print("Ask any question about the resume. Type 'quit' to exit.")

    # --- New: Create a list to store the chat history ---
    chat_history = []

    # 2. Start the chat loop
    try:
        while True:
            question = input("\n> ")
            if question.lower().strip() == 'quit':
                print("Goodbye!")
                break
            
            if not question.strip():
                continue

            print("\nThinking...")
            
            # 3. Get embedding for the question
            try:
                query_response = ollama.embeddings(
                    model=EMBEDDING_MODEL,
                    prompt=question
                )
                query_embedding = query_response["embedding"]
            except Exception as e:
                print(f"Error getting embedding for query: {e}")
                continue

            # 4. Find relevant resume chunks
            relevant_chunks = find_relevant_chunks(
                query_embedding, 
                all_embeddings, 
                text_chunks
            )
            
            # 5. Create the prompt
            context = "\n\n".join(relevant_chunks)
            
            # --- New: Build the history string ---
            history_string = "\n".join(chat_history)
            
            # --- System prompt is now loaded from config at the start of main() ---
            
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation History:\n"
                f"-----------------\n"
                f"{history_string}\n"
                f"-----------------\n\n"
                f"Resume Context:\n"
                f"-----------------\n"
                f"{context}\n"
                f"-----------------\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            # 6. Call the LLM with the prompt
            try:
                # Use streaming response
                response_stream = ollama.generate(
                    model=LLM_MODEL,
                    prompt=full_prompt,
                    stream=True
                )
                
                # --- New: Capture the full response for history ---
                full_response = ""
                for chunk in response_stream:
                    if not chunk['done']:
                        response_part = chunk['response']
                        print(response_part, end="", flush=True)
                        full_response += response_part
                print() # Newline after the full response
                
                # --- New: Add this turn to the history ---
                chat_history.append(f"User: {question}")
                chat_history.append(f"Bot: {full_response.strip()}")

            except Exception as e:
                print(f"\nAn error occurred while generating the response: {e}")

    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
