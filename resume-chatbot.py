import ollama
import fitz  # PyMuPDF
import numpy as np
import os
import json
import time
import argparse
import configparser
import sys
import re
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
    "You ARE the person in the resume context. Speak in FIRST PERSON ('I', 'my', 'me'). "
    "You are talking to a recruiter. Provide your name directly when asked - do not be evasive. "
    "Do not make up information. If unsure, say so politely. Act casual and natural."
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

def process_and_embed_resume(resume_pdf_path, personal_info_txt_path, personal_info_md_path, 
                              data_dir, embedding_model, chunk_size, chunk_overlap):
    """
    Reads, chunks, and embeds the resume and personal info file.
    Saves chunks and embeddings to disk.
    Returns text_chunks, all_embeddings, and personal_info_chunk_indices.
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
        return None, None, None
        
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

def find_relevant_chunks(query_embedding, all_embeddings, text_chunks, k, personal_info_chunk_indices=None):
    """
    Finds the top-k most similar chunks to the query.
    Always includes personal_info chunks to ensure important preferences/constraints are included.
    """
    if personal_info_chunk_indices is None:
        personal_info_chunk_indices = []
    
    similarities = []
    for emb in all_embeddings:
        sim = cosine_similarity(query_embedding, emb)
        similarities.append(sim)
    
    # Get the indices of the top-k most similar chunks
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Always include personal_info chunks (they contain important constraints/preferences)
    # Combine personal_info indices with top-k indices, removing duplicates
    combined_indices = list(set(top_k_indices.tolist() + personal_info_chunk_indices))
    
    # If we have more chunks than k, prioritize:
    # 1. Personal info chunks (always included)
    # 2. Top-k most similar chunks
    if len(combined_indices) > k:
        # Start with personal_info chunks
        result_indices = personal_info_chunk_indices.copy()
        
        # Add top-k chunks that aren't already in personal_info
        for idx in top_k_indices:
            if idx not in result_indices and len(result_indices) < k:
                result_indices.append(idx)
        
        # If we still have room and personal_info chunks, fill with remaining top-k
        remaining_slots = k - len(result_indices)
        if remaining_slots > 0:
            for idx in top_k_indices:
                if idx not in result_indices:
                    result_indices.append(idx)
                    remaining_slots -= 1
                    if remaining_slots == 0:
                        break
    else:
        result_indices = combined_indices
    
    # Return the text of those chunks
    return [text_chunks[i] for i in result_indices]

class RecruiterInfoTracker:
    """
    Tracks information about the recruiter/interviewer and job opportunity.
    """
    def __init__(self, applicant_name=None):
        self.recruiter_name = None
        self.contact_info = None
        self.job_synopsis = None
        self.benefits = None
        self.salary = None
        self._conversation_text = ""
        self.applicant_name = applicant_name  # Store applicant name to filter it out
    
    def add_conversation(self, user_message, bot_response):
        """Add a conversation exchange to track."""
        # Note: User is the recruiter/interviewer, Bot is the applicant
        self._conversation_text += f"Recruiter/Interviewer: {user_message}\nApplicant: {bot_response}\n\n"
    
    def extract_info(self, llm_model):
        """
        Uses LLM to extract recruiter and job information from conversation.
        Returns a dict with extracted information.
        """
        if not self._conversation_text.strip():
            return {}
        
        # Build prompt with applicant name exclusion if known
        applicant_exclusion = ""
        if self.applicant_name:
            applicant_exclusion = f"IMPORTANT: The applicant's name is '{self.applicant_name}'. Do NOT extract this name. Only extract the recruiter/interviewer's name (the person talking TO the applicant). "
        
        extraction_prompt = (
            "Extract the following information from this conversation between a job applicant and a recruiter/interviewer. "
            f"{applicant_exclusion}"
            "CRITICAL: Only extract information that is EXPLICITLY MENTIONED in the conversation. "
            "Do NOT infer, assume, or add information that was not directly stated. "
            "Do NOT add common benefits or standard information unless it was specifically mentioned. "
            "If information is not explicitly mentioned, you MUST return 'Not mentioned' for that field. "
            "\n"
            "Extract the name of the RECRUITER/INTERVIEWER (the person talking TO the applicant), NOT the applicant's name. "
            "Only extract the name of the person representing the company who is conducting the interview. "
            "\n"
            "For benefits: Only list benefits that were EXPLICITLY MENTIONED in the conversation. "
            "Do NOT add standard benefits like 'health insurance' or '401k' unless they were specifically discussed. "
            "If no benefits were mentioned, return 'Not mentioned'. "
            "\n"
            "Return ONLY a JSON object with these exact keys:\n"
            "{\n"
            '  "recruiter_name": "name of the recruiter/interviewer (NOT the applicant) or Not mentioned",\n'
            '  "contact_info": "email/phone of the recruiter/interviewer or Not mentioned",\n'
            '  "job_synopsis": "brief job description or Not mentioned",\n'
            '  "benefits": "ONLY benefits explicitly mentioned in conversation or Not mentioned",\n'
            '  "salary": "salary/compensation explicitly mentioned or Not mentioned"\n'
            "}\n\n"
            f"Conversation:\n{self._conversation_text}\n\n"
            "Remember: Only extract what was EXPLICITLY STATED. Do not infer or add information. "
            "JSON only, no other text:"
        )
        
        try:
            response = ollama.generate(
                model=llm_model,
                prompt=extraction_prompt,
                stream=False
            )
            response_text = response['response'].strip()
            
            # Try to extract JSON from response (handle nested braces)
            # Find the first { and match to the last }
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    try:
                        extracted = json.loads(json_str)
                    except json.JSONDecodeError:
                        extracted = {}
                else:
                    extracted = {}
            else:
                extracted = {}
            
            if extracted:
                
                # Update fields if new information is found
                if extracted.get('recruiter_name') and extracted['recruiter_name'] != 'Not mentioned':
                    # Filter out applicant's name if it matches
                    extracted_name = extracted['recruiter_name']
                    if self.applicant_name and extracted_name.lower() == self.applicant_name.lower():
                        # Skip - this is the applicant's name, not the recruiter's
                        pass
                    elif not self.recruiter_name:
                        self.recruiter_name = extracted_name
                
                if extracted.get('contact_info') and extracted['contact_info'] != 'Not mentioned':
                    if not self.contact_info:
                        self.contact_info = extracted['contact_info']
                
                if extracted.get('job_synopsis') and extracted['job_synopsis'] != 'Not mentioned':
                    if not self.job_synopsis:
                        self.job_synopsis = extracted['job_synopsis']
                
                if extracted.get('benefits') and extracted['benefits'] != 'Not mentioned':
                    # Validate that benefits were actually mentioned in conversation
                    benefits_text = extracted['benefits'].lower()
                    conversation_lower = self._conversation_text.lower()
                    # Check if any key benefit words from extraction appear in conversation
                    # This is a simple validation - if benefits contain words not in conversation, be cautious
                    benefit_words = set(benefits_text.split())
                    conversation_words = set(conversation_lower.split())
                    # Allow some common connecting words
                    common_words = {'and', 'or', 'the', 'a', 'an', 'with', 'including', 'plus', 'also'}
                    benefit_keywords = benefit_words - common_words
                    # If most benefit keywords appear in conversation, accept it
                    if benefit_keywords:
                        matches = sum(1 for word in benefit_keywords if word in conversation_words)
                        # Require at least 50% of keywords to match, or if it's a short phrase, all must match
                        if matches >= len(benefit_keywords) * 0.5 or len(benefit_keywords) <= 3:
                            if not self.benefits:
                                self.benefits = extracted['benefits']
                        # If validation fails, don't update benefits
                
                if extracted.get('salary') and extracted['salary'] != 'Not mentioned':
                    if not self.salary:
                        self.salary = extracted['salary']
                
                return extracted
        except Exception as e:
            # Silently fail - extraction is optional
            pass
        
        return {}
    
    def format_for_log(self):
        """Format the tracked information for log file header."""
        lines = []
        lines.append("RECRUITER/INTERVIEWER INFORMATION:")
        lines.append("-" * 80)
        lines.append(f"Name: {self.recruiter_name or 'Not yet mentioned'}")
        lines.append(f"Contact Info: {self.contact_info or 'Not yet mentioned'}")
        lines.append("")
        lines.append("JOB OPPORTUNITY:")
        lines.append("-" * 80)
        lines.append(f"Job Synopsis: {self.job_synopsis or 'Not yet mentioned'}")
        lines.append(f"Benefits: {self.benefits or 'Not yet mentioned'}")
        lines.append(f"Salary: {self.salary or 'Not yet mentioned'}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        return "\n".join(lines)
    
    def has_any_info(self):
        """Check if any information has been extracted."""
        return any([self.recruiter_name, self.contact_info, self.job_synopsis, 
                   self.benefits, self.salary])

def initialize_logging(logs_dir):
    """
    Creates the logs directory and opens a timestamped log file for this session.
    Returns the file handle, filename, and log filepath.
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
    log_file.write("RECRUITER/INTERVIEWER INFORMATION:\n")
    log_file.write("-" * 80 + "\n")
    log_file.write("Name: Not yet mentioned\n")
    log_file.write("Contact Info: Not yet mentioned\n\n")
    log_file.write("JOB OPPORTUNITY:\n")
    log_file.write("-" * 80 + "\n")
    log_file.write("Job Synopsis: Not yet mentioned\n")
    log_file.write("Benefits: Not yet mentioned\n")
    log_file.write("Salary: Not yet mentioned\n")
    log_file.write("\n" + "=" * 80 + "\n\n")
    log_file.flush()
    
    return log_file, log_filename, log_filepath

def update_log_header(log_filepath, recruiter_tracker):
    """
    Updates the header section of the log file with extracted recruiter/job information.
    """
    try:
        # Read the entire file
        with open(log_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find where the recruiter info section starts and ends
        # We'll replace from "RECRUITER/INTERVIEWER INFORMATION:" to the "=" line before conversation
        pattern = r'(RECRUITER/INTERVIEWER INFORMATION:.*?=+\n\n)'
        
        # Generate new header section
        new_header = recruiter_tracker.format_for_log()
        
        # Replace the old header section
        new_content = re.sub(pattern, new_header, content, flags=re.DOTALL)
        
        # Write back to file
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except Exception as e:
        # Silently fail - header update is optional
        pass

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
    text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name = None, None, None, None
    if not args.reprocess:
        text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name = load_data_from_disk(data_dir)
    
    if text_chunks is None or all_embeddings is None:
        if args.reprocess:
            print("\n--reprocess flag detected. Forcing file processing...")
        text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name = process_and_embed_resume(
            resume_pdf_path, personal_info_txt_path, personal_info_md_path,
            data_dir, embedding_model, chunk_size, chunk_overlap
        )
        if text_chunks is None:
            print("Exiting due to error.")
            return
    
    # Ensure personal_info_chunk_indices is a list (for backward compatibility)
    if personal_info_chunk_indices is None:
        personal_info_chunk_indices = []

    end_time = time.time()
    print(f"\n--- Ready to chat! (Setup took {end_time - start_time:.2f}s) ---")
    print("Ask any question about the resume. Type 'quit' to exit.")
    print(f"Context management: Max history tokens={max_history_tokens}, Min recent messages={min_recent_messages}")

    # --- Initialize logging ---
    log_file, log_filename, log_filepath = initialize_logging(logs_dir)
    print(f"Conversation log: {log_filename}")
    
    # --- Create recruiter info tracker ---
    recruiter_tracker = RecruiterInfoTracker(applicant_name=extracted_name)
    
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

            # 4. Find relevant resume chunks (always includes personal_info chunks)
            relevant_chunks = find_relevant_chunks(
                query_embedding, 
                all_embeddings, 
                text_chunks,
                top_k_chunks,
                personal_info_chunk_indices
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
            
            # Build the prompt with explicit name if available
            name_reminder = ""
            if extracted_name:
                name_reminder = f"IMPORTANT: Your name is {extracted_name}. When asked for your name, say '{extracted_name}' directly - do not be evasive.\n\n"
            
            full_prompt += (
                f"Your Background (this is information about YOU - the person speaking):\n"
                f"-----------------\n"
                f"{resume_context}\n"
                f"-----------------\n\n"
                f"{name_reminder}"
                f"Remember: You ARE the person described above. Speak in first person about your own background and experience.\n\n"
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
                
                # Track recruiter/job information
                recruiter_tracker.add_conversation(question, full_response.strip())
                
                # Periodically extract and update info (every 3 exchanges to avoid too many LLM calls)
                if len(recruiter_tracker._conversation_text.split('\n\n')) % 6 == 0:
                    recruiter_tracker.extract_info(llm_model)
                    if recruiter_tracker.has_any_info():
                        update_log_header(log_filepath, recruiter_tracker)
                
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
            # Final extraction attempt before closing
            recruiter_tracker.extract_info(llm_model)
            if recruiter_tracker.has_any_info():
                update_log_header(log_filepath, recruiter_tracker)
            
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
