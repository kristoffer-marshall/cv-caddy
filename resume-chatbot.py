import ollama
import fitz  # PyMuPDF
import numpy as np
import os
import json
import time
import argparse
import configparser
import sys
from datetime import datetime

# --- CONFIGURATION ---
# Only CONFIG_FILE remains here as it's needed to read the config itself
CONFIG_FILE = "config.ini"

# Default values (used as fallbacks if config.ini is missing or incomplete)
DEFAULT_RESUME_PDF_PATH = "YOUR_RESUME.pdf"
DEFAULT_PERSONAL_INFO_TXT_PATH = "personal_info.txt"
DEFAULT_PERSONAL_INFO_MD_PATH = "personal_info.md"
DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_MAX_HISTORY_TOKENS = 2000
DEFAULT_MIN_RECENT_MESSAGES = 6
DEFAULT_SUMMARY_THRESHOLD = 0.8
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K_CHUNKS = 3
DEFAULT_SYSTEM_PROMPT = (
    "You are a professional vulnerability management engineer who is talking to a recruiter or hiring manager. "
    "Do not make up information. If the answer is not in the context, "
    "politely state that you are unsure, but can get back to them later. "
    "Use the knowledge from your resume, but do no reference your resume in conversation. Act casual."
)
# ---------------------


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

def process_and_embed_resume(resume_pdf_path, personal_info_txt_path, personal_info_md_path, 
                              data_dir, embedding_model, chunk_size, chunk_overlap):
    """
    Reads, chunks, and embeds the resume and personal info file.
    Saves chunks and embeddings to disk.
    """
    chunks_file = os.path.join(data_dir, "chunks.json")
    embeddings_file = os.path.join(data_dir, "embeddings.npy")
    
    print("Checking for context files...")
    full_text = ""
    
    # 1. Read PDF
    if os.path.exists(resume_pdf_path):
        try:
            doc = fitz.open(resume_pdf_path)
            for page in doc:
                full_text += page.get_text() + "\n" # Add newline separator
            
            print(f"Successfully read {len(doc)} pages from PDF '{resume_pdf_path}'.")
            doc.close()
        except Exception as e:
            print(f"Error reading PDF '{resume_pdf_path}': {e}")
            # We can continue if the other file exists
    else:
        print(f"Warning: Resume file not found at '{resume_pdf_path}'.")

    # 2. Read personal info (MD or TXT)
    personal_file_found = False
    # Prioritize Markdown file
    if os.path.exists(personal_info_md_path):
        try:
            with open(personal_info_md_path, "r") as f:
                full_text += f.read()
            print(f"Successfully read markdown file '{personal_info_md_path}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading markdown file '{personal_info_md_path}': {e}")
    
    # If no MD file, check for TXT file as fallback
    if not personal_file_found and os.path.exists(personal_info_txt_path):
        try:
            with open(personal_info_txt_path, "r") as f:
                full_text += f.read()
            print(f"Successfully read text file '{personal_info_txt_path}'.")
            personal_file_found = True
        except Exception as e:
            print(f"Error reading text file '{personal_info_txt_path}': {e}")

    if not personal_file_found:
        print(f"Info: Personal info file not found at '{personal_info_md_path}' or '{personal_info_txt_path}'. This is optional.")

    # 3. Check if we have any text at all
    if not full_text.strip():
        print("Error: No text found from resume or personal info file.")
        print(f"Please make sure '{resume_pdf_path}' or '{personal_info_md_path}' or '{personal_info_txt_path}' exists.")
        return None, None
        
    print(f"Processing combined text...")
    
    # 4. Split into chunks
    text_chunks = split_text_into_chunks(full_text, chunk_size, chunk_overlap)
    print(f"Split text into {len(text_chunks)} chunks.")

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
        
        print("Successfully created and saved chunks and embeddings.")
        return text_chunks, np_embeddings

    except Exception as e:
        print(f"Error creating embeddings: {e}")
        print("Is Ollama running? Try 'ollama serve' in another terminal.")
        return None, None

def load_data_from_disk(data_dir):
    """
    Loads processed chunks and embeddings from the data directory.
    """
    chunks_file = os.path.join(data_dir, "chunks.json")
    embeddings_file = os.path.join(data_dir, "embeddings.npy")
    
    print(f"Loading existing data from '{data_dir}' directory...")
    if not os.path.exists(chunks_file) or not os.path.exists(embeddings_file):
        return None, None

    try:
        with open(chunks_file, "r") as f:
            text_chunks = json.load(f)
        
        all_embeddings = np.load(embeddings_file)
        
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

def estimate_tokens(text):
    """
    Estimates token count from text. Uses a simple approximation:
    ~4 characters per token for English text (conservative estimate).
    """
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English
    # This is conservative to ensure we stay within limits
    return len(text) // 4

class ContextManager:
    """
    Manages conversation history with context window limits.
    Ensures resume context is always preserved while managing history size.
    """
    
    def __init__(self, max_history_tokens, min_recent_messages, summary_threshold, llm_model):
        self.max_history_tokens = max_history_tokens
        self.min_recent_messages = min_recent_messages
        self.summary_threshold = summary_threshold
        self.llm_model = llm_model
        self.history = []  # List of (role, message) tuples
        self.summary = None  # Summarized older conversation
        
    def add_exchange(self, user_message, bot_message):
        """
        Add a user-bot exchange to the history.
        """
        self.history.append(("User", user_message))
        self.history.append(("Bot", bot_message))
        self._manage_context()
    
    def _manage_context(self):
        """
        Manages context window by summarizing old messages when needed.
        Always preserves recent messages to maintain conversation flow.
        """
        # Calculate current history size
        history_text = self._format_history(self.history)
        history_tokens = estimate_tokens(history_text)
        
        # Check if we need to manage context
        if history_tokens <= self.max_history_tokens * self.summary_threshold:
            return  # No action needed
        
        # We need to summarize. Keep recent messages, summarize older ones
        recent_count = max(self.min_recent_messages, len(self.history) // 4)
        recent_messages = self.history[-recent_count:]
        old_messages = self.history[:-recent_count]
        
        if not old_messages:
            # Can't reduce further without losing recent context
            return
        
        # Create or update summary of old messages
        old_history_text = self._format_history(old_messages)
        if self.summary:
            # Combine existing summary with new old messages
            summary_prompt = (
                f"Previous conversation summary:\n{self.summary}\n\n"
                f"Additional conversation to add to summary:\n{old_history_text}\n\n"
                f"Create a concise summary that captures the key points from both the previous summary "
                f"and the additional conversation. Focus on facts, decisions, and important context. "
                f"Keep it brief (under 200 words)."
            )
        else:
            summary_prompt = (
                f"Summarize the following conversation, focusing on key facts, decisions, "
                f"and important context. Keep it brief (under 200 words):\n\n{old_history_text}"
            )
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=summary_prompt,
                stream=False
            )
            self.summary = response['response'].strip()
            
            # Replace old messages with summary
            self.history = recent_messages
            
        except Exception as e:
            print(f"\nWarning: Could not summarize conversation history: {e}")
            # Fallback: just truncate to recent messages
            self.history = recent_messages
    
    def _format_history(self, messages):
        """
        Formats a list of (role, message) tuples into a history string.
        """
        return "\n".join([f"{role}: {message}" for role, message in messages])
    
    def get_formatted_history(self):
        """
        Returns the formatted conversation history, including summary if present.
        """
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary: {self.summary}]")
        if self.history:
            parts.append(self._format_history(self.history))
        return "\n".join(parts) if parts else ""
    
    def get_history_token_count(self):
        """
        Returns estimated token count of current history.
        """
        history_text = self.get_formatted_history()
        return estimate_tokens(history_text)

def find_relevant_chunks(query_embedding, all_embeddings, text_chunks, k):
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

def initialize_logging(logs_dir):
    """
    Creates the logs directory and opens a timestamped log file for this session.
    Returns the file handle and filename.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"conversation_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)
    
    # Open log file in append mode (though it will be new)
    log_file = open(log_filepath, 'a', encoding='utf-8')
    
    # Write session header
    session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{'='*80}\n")
    log_file.write(f"Conversation Session Started: {session_start}\n")
    log_file.write(f"{'='*80}\n\n")
    log_file.flush()
    
    return log_file, log_filename

def log_exchange(log_file, user_question, bot_response):
    """
    Logs a user-bot exchange to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_file.write(f"[{timestamp}] User:\n{user_question}\n\n")
    log_file.write(f"[{timestamp}] Bot:\n{bot_response}\n\n")
    log_file.write(f"{'-'*80}\n\n")
    log_file.flush()

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
    
    # --- Config Parsing ---
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"'{CONFIG_FILE}' not found. Creating a default config file.")
        try:
            # Create default config with all sections
            config['Files'] = {
                'ResumePdfPath': DEFAULT_RESUME_PDF_PATH,
                'PersonalInfoTxtPath': DEFAULT_PERSONAL_INFO_TXT_PATH,
                'PersonalInfoMdPath': DEFAULT_PERSONAL_INFO_MD_PATH,
                'DataDir': DEFAULT_DATA_DIR,
                'LogsDir': DEFAULT_LOGS_DIR
            }
            config['Models'] = {
                'LlmModel': DEFAULT_LLM_MODEL,
                'EmbeddingModel': DEFAULT_EMBEDDING_MODEL
            }
            config['Context'] = {
                'MaxHistoryTokens': str(DEFAULT_MAX_HISTORY_TOKENS),
                'MinRecentMessages': str(DEFAULT_MIN_RECENT_MESSAGES),
                'SummaryThreshold': str(DEFAULT_SUMMARY_THRESHOLD)
            }
            config['RAG'] = {
                'ChunkSize': str(DEFAULT_CHUNK_SIZE),
                'ChunkOverlap': str(DEFAULT_CHUNK_OVERLAP),
                'TopKChunks': str(DEFAULT_TOP_K_CHUNKS)
            }
            config['Chatbot'] = {'SystemPrompt': DEFAULT_SYSTEM_PROMPT}
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            print(f"Default '{CONFIG_FILE}' created successfully.")
        except Exception as e:
            print(f"Error creating default config file: {e}")
            print("Using hard-coded defaults.")
    else:
        try:
            config.read(CONFIG_FILE)
        except Exception as e:
            print(f"Error reading '{CONFIG_FILE}': {e}. Using defaults.")

    # Load all configuration values with fallbacks
    # Files section
    resume_pdf_path = config.get('Files', 'ResumePdfPath', fallback=DEFAULT_RESUME_PDF_PATH)
    personal_info_txt_path = config.get('Files', 'PersonalInfoTxtPath', fallback=DEFAULT_PERSONAL_INFO_TXT_PATH)
    personal_info_md_path = config.get('Files', 'PersonalInfoMdPath', fallback=DEFAULT_PERSONAL_INFO_MD_PATH)
    data_dir = config.get('Files', 'DataDir', fallback=DEFAULT_DATA_DIR)
    logs_dir = config.get('Files', 'LogsDir', fallback=DEFAULT_LOGS_DIR)
    
    # Models section
    llm_model = config.get('Models', 'LlmModel', fallback=DEFAULT_LLM_MODEL)
    embedding_model = config.get('Models', 'EmbeddingModel', fallback=DEFAULT_EMBEDDING_MODEL)
    
    # Context section
    max_history_tokens = config.getint('Context', 'MaxHistoryTokens', fallback=DEFAULT_MAX_HISTORY_TOKENS)
    min_recent_messages = config.getint('Context', 'MinRecentMessages', fallback=DEFAULT_MIN_RECENT_MESSAGES)
    summary_threshold = config.getfloat('Context', 'SummaryThreshold', fallback=DEFAULT_SUMMARY_THRESHOLD)
    
    # RAG section
    chunk_size = config.getint('RAG', 'ChunkSize', fallback=DEFAULT_CHUNK_SIZE)
    chunk_overlap = config.getint('RAG', 'ChunkOverlap', fallback=DEFAULT_CHUNK_OVERLAP)
    top_k_chunks = config.getint('RAG', 'TopKChunks', fallback=DEFAULT_TOP_K_CHUNKS)
    
    # Chatbot section
    system_prompt = config.get('Chatbot', 'SystemPrompt', fallback=DEFAULT_SYSTEM_PROMPT)
    
    start_time = time.time()
    
    # 1. Load or create data (with reprocess logic)
    text_chunks, all_embeddings = None, None
    if not args.reprocess:
        text_chunks, all_embeddings = load_data_from_disk(data_dir)
    
    if text_chunks is None or all_embeddings is None:
        if args.reprocess:
            print("\n--reprocess flag detected. Forcing file processing...")
        text_chunks, all_embeddings = process_and_embed_resume(
            resume_pdf_path, personal_info_txt_path, personal_info_md_path,
            data_dir, embedding_model, chunk_size, chunk_overlap
        )
        if text_chunks is None:
            print("Exiting due to error.")
            return

    end_time = time.time()
    print(f"\n--- Ready to chat! (Setup took {end_time - start_time:.2f}s) ---")
    print("Ask any question about the resume. Type 'quit' to exit.")
    print(f"Context management: Max history tokens={max_history_tokens}, Min recent messages={min_recent_messages}")

    # --- Initialize logging ---
    log_file, log_filename = initialize_logging(logs_dir)
    print(f"Conversation log: {log_filename}")
    
    # --- Create context manager to handle conversation history ---
    context_manager = ContextManager(
        max_history_tokens=max_history_tokens,
        min_recent_messages=min_recent_messages,
        summary_threshold=summary_threshold,
        llm_model=llm_model
    )

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
                    model=embedding_model,
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
                text_chunks,
                top_k_chunks
            )
            
            # 5. Create the prompt
            # Resume context is always included - this is the source of truth
            resume_context = "\n\n".join(relevant_chunks)
            
            # Get managed conversation history
            history_string = context_manager.get_formatted_history()
            
            # --- System prompt is now loaded from config at the start of main() ---
            
            # Build prompt with clear separation between history and resume context
            # Resume context is always preserved to prevent hallucination
            full_prompt = (
                f"{system_prompt}\n\n"
            )
            
            # Only include history section if there's actual history
            if history_string.strip():
                full_prompt += (
                    f"Conversation History:\n"
                    f"-----------------\n"
                    f"{history_string}\n"
                    f"-----------------\n\n"
                )
            
            full_prompt += (
                f"Resume Context (source of truth - always refer to this for factual information):\n"
                f"-----------------\n"
                f"{resume_context}\n"
                f"-----------------\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            # 6. Call the LLM with the prompt
            try:
                # Use streaming response
                response_stream = ollama.generate(
                    model=llm_model,
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
                
                # Add this exchange to context manager (handles windowing automatically)
                context_manager.add_exchange(question, full_response.strip())
                
                # Log this exchange
                log_exchange(log_file, question, full_response.strip())
                
                # Optional: Show context stats (can be removed or made configurable)
                history_tokens = context_manager.get_history_token_count()
                if history_tokens > max_history_tokens * 0.7:
                    print(f"\n[Context: {history_tokens}/{max_history_tokens} tokens used]")

            except Exception as e:
                print(f"\nAn error occurred while generating the response: {e}")
                # Log the error
                try:
                    error_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{error_timestamp}] Error: {str(e)}\n\n")
                    log_file.flush()
                except:
                    pass  # Ignore logging errors

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        # Always close the log file, whether normal exit or interrupt
        try:
            session_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{'='*80}\n")
            # Check if we're exiting due to KeyboardInterrupt
            exc_type = sys.exc_info()[0]
            if exc_type is KeyboardInterrupt:
                log_file.write(f"Conversation Session Ended (Interrupted): {session_end}\n")
            else:
                log_file.write(f"Conversation Session Ended: {session_end}\n")
            log_file.write(f"{'='*80}\n")
            log_file.close()
        except:
            pass  # Ignore errors when closing log file

if __name__ == "__main__":
    main()
