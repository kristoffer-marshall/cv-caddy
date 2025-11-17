"""
Main entry point for the resume chatbot application.
"""
import os
import sys
import time
import argparse
import configparser
import ollama
from datetime import datetime

from config import (
    CONFIG_FILE,
    DEFAULT_RESUME_PDF_PATH,
    DEFAULT_PERSONAL_INFO_TXT_PATH,
    DEFAULT_PERSONAL_INFO_MD_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_HISTORY_TOKENS,
    DEFAULT_MIN_RECENT_MESSAGES,
    DEFAULT_SUMMARY_THRESHOLD,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K_CHUNKS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P
)
from resume_processor import process_and_embed_resume, load_data_from_disk
from rag import find_relevant_chunks
from context_manager import ContextManager
from recruiter_tracker import RecruiterInfoTracker
from logging_utils import initialize_logging, update_log_header, log_exchange


def main():
    """
    Main function to run the chatbot.
    """
    
    # --- Argument Parsing ---
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
                'EmbeddingModel': DEFAULT_EMBEDDING_MODEL,
                'Temperature': str(DEFAULT_TEMPERATURE),
                'TopP': str(DEFAULT_TOP_P)
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
    temperature = config.getfloat('Models', 'Temperature', fallback=DEFAULT_TEMPERATURE)
    top_p = config.getfloat('Models', 'TopP', fallback=DEFAULT_TOP_P)
    
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
            
            # Check if question is about salary and add specific reminder
            salary_reminder = ""
            question_lower = question.lower()
            if any(word in question_lower for word in ['salary', 'compensation', 'pay', 'wage', 'earn', 'requirement', 'expect', 'ballpark']):
                salary_reminder = (
                    "CRITICAL REMINDER: Do NOT provide your salary requirement, ballpark figure, or any specific salary amount. "
                    "If asked about salary, politely deflect by saying you'd prefer to discuss compensation after learning more about the role, "
                    "or that you're open to negotiation. Do not give specific numbers.\n\n"
                )
            
            full_prompt += (
                f"Your Background:\n"
                f"{resume_context}\n\n"
                f"{name_reminder}"
                f"{salary_reminder}"
                f"Now, respond naturally to this question as if you're having a real conversation:\n\n"
                f"{question}\n\n"
            )

            # 6. Call the LLM with the prompt
            try:
                # Use streaming response with temperature and top_p for more natural responses
                response_stream = ollama.generate(
                    model=llm_model,
                    prompt=full_prompt,
                    stream=True,
                    options={
                        'temperature': temperature,
                        'top_p': top_p
                    }
                )
                
                # Capture the full response for history
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

