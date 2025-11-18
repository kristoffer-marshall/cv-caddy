"""
Main entry point for the resume chatbot application.
"""
import os
import sys
import time
import random
import argparse
import configparser
import ollama
import threading
import socket
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


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_BLUE = '\033[104m'
    BG_GRAY = '\033[100m'


class TypingIndicator:
    """Manages animated typing indicator for SMS mode."""
    
    def __init__(self):
        self.is_typing = False
        self.stop_event = threading.Event()
        self.thread = None
    
    def start(self, initial_delay=0):
        """
        Start the typing indicator animation.
        
        Args:
            initial_delay: Delay in seconds before starting the animation (for reading/processing time)
        """
        if initial_delay > 0:
            # Wait for the initial delay before starting
            time.sleep(initial_delay)
        
        self.is_typing = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the typing indicator animation."""
        self.is_typing = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear the typing indicator line
        print('\r' + ' ' * 50 + '\r', end='', flush=True)
    
    def _animate(self):
        """Animate the typing indicator with random pauses."""
        dots = ['.', '..', '...']
        dot_index = 0
        
        while not self.stop_event.is_set():
            # Display typing indicator
            indicator = f"{Colors.GRAY}{dots[dot_index % len(dots)]}{Colors.RESET}"
            print(f'\r{indicator}', end='', flush=True)
            
            # Random pause between 0.3 and 0.8 seconds
            pause = random.uniform(0.3, 0.8)
            if self.stop_event.wait(pause):
                break
            
            dot_index += 1
            
            # Randomly pause typing (like human hesitation)
            if random.random() < 0.15:  # 15% chance of pausing
                pause_time = random.uniform(0.5, 1.5)
                print('\r' + ' ' * 10 + '\r', end='', flush=True)
                if self.stop_event.wait(pause_time):
                    break


def get_first_name(full_name):
    """
    Extract the first name from a full name.
    
    Args:
        full_name: Full name string (e.g., "John Doe" or "John Michael Doe")
    
    Returns:
        First name string, or None if full_name is None/empty
    """
    if not full_name:
        return None
    # Split by space and take the first part
    name_parts = full_name.strip().split()
    return name_parts[0] if name_parts else None


def display_sms_message(message, is_user=False, sender_name=None):
    """
    Display a message in SMS/RCS style with colors.
    
    Args:
        message: The message text to display
        is_user: True if this is a user message, False if bot message
        sender_name: Name to display for bot messages (defaults to "Them" if not provided)
    """
    if is_user:
        # User messages: blue text, bold
        print(f"{Colors.BLUE}{Colors.BOLD}You:{Colors.RESET} {message}")
    else:
        # Bot messages: gray text, subtle
        display_name = sender_name if sender_name else "Them"
        print(f"{Colors.GRAY}{display_name}:{Colors.RESET} {message}")


def display_sms_response(response_text, typing_indicator, sender_name=None):
    """
    Display bot response in SMS style with typing simulation.
    
    Args:
        response_text: The full response text
        typing_indicator: TypingIndicator instance
        sender_name: Name to display for bot messages (defaults to "Them" if not provided)
    """
    # Stop typing indicator
    typing_indicator.stop()
    
    # Simulate human typing delay before showing message
    # Random delay between 0.2 and 0.8 seconds
    typing_delay = random.uniform(0.2, 0.8)
    time.sleep(typing_delay)
    
    # Display the message
    display_sms_message(response_text, is_user=False, sender_name=sender_name)


def detect_quit_intent(user_input):
    """
    Detects if the user wants to quit/end the conversation.
    Returns True if quit intent is detected, False otherwise.
    """
    if not user_input:
        return False
    
    user_lower = user_input.lower().strip()
    
    # Direct quit commands
    quit_commands = ['quit', 'exit', 'bye', 'goodbye', 'good bye', 'see ya', 'see you', 
                     'talk later', 'gotta go', 'have to go', 'i have to go', 'i gotta go',
                     'thanks bye', 'thanks goodbye', 'thank you bye', 'thank you goodbye',
                     'that\'s all', "that's all", 'that is all', 'all done', 'we\'re done',
                     "we're done", 'we are done', 'i\'m done', "i'm done", 'i am done',
                     'end conversation', 'end chat', 'close', 'done', 'finished']
    
    # Check for exact matches
    if user_lower in quit_commands:
        return True
    
    # Check for phrases that indicate leaving
    quit_phrases = [
        'have to go', 'gotta go', 'got to go', 'need to go', 'should go',
        'talk to you later', 'speak with you later', 'catch you later',
        'thanks for', 'thank you for', 'appreciate your time',
        'nice talking', 'nice chatting', 'nice speaking',
        'take care', 'have a good', 'have a great', 'have a nice'
    ]
    
    for phrase in quit_phrases:
        if phrase in user_lower:
            return True
    
    return False


def generate_goodbye(llm_model, system_prompt, history_string, extracted_name, temperature, top_p):
    """
    Generates a natural goodbye message using the LLM.
    """
    goodbye_prompt = (
        f"{system_prompt}\n\n"
    )
    
    if history_string.strip():
        goodbye_prompt += (
            f"Conversation History:\n"
            f"-----------------\n"
            f"{history_string}\n"
            f"-----------------\n\n"
        )
    
    name_context = ""
    if extracted_name:
        name_context = f"Your name is {extracted_name}. "
    
    goodbye_prompt += (
        f"{name_context}"
        f"The recruiter is ending the conversation. Give them a natural, professional, "
        f"and friendly goodbye. Thank them for their time, express interest in next steps, "
        f"and keep it brief (1-2 sentences). Be warm but professional. "
        f"Do NOT wrap your response in quotes - respond directly as if speaking.\n\n"
    )
    
    try:
        response = ollama.generate(
            model=llm_model,
            prompt=goodbye_prompt,
            stream=False,
            options={
                'temperature': temperature,
                'top_p': top_p
            }
        )
        goodbye_text = response['response'].strip()
        # Remove surrounding quotes if present
        if goodbye_text.startswith('"') and goodbye_text.endswith('"'):
            goodbye_text = goodbye_text[1:-1].strip()
        elif goodbye_text.startswith("'") and goodbye_text.endswith("'"):
            goodbye_text = goodbye_text[1:-1].strip()
        return goodbye_text
    except Exception as e:
        # Fallback to a simple goodbye if LLM fails
        return "Thank you so much for your time! I really appreciate the opportunity to speak with you. Looking forward to hearing about next steps."


def handle_telnet_client(client_socket, client_address, llm_model, embedding_model, system_prompt,
                         text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                         context_manager, log_file, top_k_chunks, temperature, top_p, first_name):
    """
    Handle a single telnet client connection.
    
    Args:
        client_socket: The socket connection to the client
        client_address: Tuple of (host, port) for the client
        llm_model: LLM model name
        embedding_model: Embedding model name
        system_prompt: System prompt for the chatbot
        text_chunks: List of text chunks
        all_embeddings: Numpy array of embeddings
        personal_info_chunk_indices: List of indices for personal info chunks
        extracted_name: Extracted name from resume
        context_manager: ContextManager instance
        log_file: Log file handle
        top_k_chunks: Number of top chunks to retrieve
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        first_name: First name for display
    """
    try:
        # Send welcome message
        welcome_msg = "\r\nWelcome to the Resume Chatbot!\r\nType your questions below. Type 'quit' to exit.\r\n\r\n"
        client_socket.sendall(welcome_msg.encode('utf-8'))
        
        # Create a new context manager for this client
        client_context = ContextManager(
            max_history_tokens=context_manager.max_history_tokens,
            min_recent_messages=context_manager.min_recent_messages,
            summary_threshold=context_manager.summary_threshold,
            llm_model=context_manager.llm_model
        )
        
        # Create a recruiter tracker for this client
        client_recruiter_tracker = RecruiterInfoTracker(applicant_name=extracted_name)
        
        while True:
            # Send prompt
            client_socket.sendall(b"> ")
            
            # Receive input from client
            question = ""
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        return  # Client disconnected
                    
                    # Decode and handle line endings
                    question += data.decode('utf-8', errors='ignore')
                    if '\n' in question or '\r' in question:
                        # Clean up line endings
                        question = question.strip().replace('\r', '').replace('\n', '')
                        break
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error receiving data from {client_address}: {e}")
                    return
            
            if not question:
                continue
            
            # Check for quit intent
            if detect_quit_intent(question):
                history_string = client_context.get_formatted_history()
                goodbye_message = generate_goodbye(
                    llm_model, system_prompt, history_string, extracted_name, temperature, top_p
                )
                client_socket.sendall(f"\r\n{goodbye_message}\r\n".encode('utf-8'))
                log_exchange(log_file, question, goodbye_message)
                break
            
            # Send "Thinking..." message
            client_socket.sendall(b"\r\nThinking...\r\n")
            
            # Get embedding for the question
            try:
                query_response = ollama.embeddings(
                    model=embedding_model,
                    prompt=question
                )
                query_embedding = query_response["embedding"]
            except Exception as e:
                error_msg = f"\r\nError getting embedding for query: {e}\r\n"
                client_socket.sendall(error_msg.encode('utf-8'))
                continue
            
            # Find relevant resume chunks
            relevant_chunks = find_relevant_chunks(
                query_embedding,
                all_embeddings,
                text_chunks,
                top_k_chunks,
                personal_info_chunk_indices
            )
            
            # Create the prompt
            resume_context = "\n\n".join(relevant_chunks)
            history_string = client_context.get_formatted_history()
            
            full_prompt = f"{system_prompt}\n\n"
            
            if history_string.strip():
                full_prompt += (
                    f"Conversation History:\n"
                    f"-----------------\n"
                    f"{history_string}\n"
                    f"-----------------\n\n"
                )
            
            name_reminder = ""
            if extracted_name:
                name_reminder = f"IMPORTANT: Your name is {extracted_name}. When asked for your name, say '{extracted_name}' directly - do not be evasive.\n\n"
            
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
            
            # Call the LLM
            try:
                response_stream = ollama.generate(
                    model=llm_model,
                    prompt=full_prompt,
                    stream=True,
                    options={
                        'temperature': temperature,
                        'top_p': top_p
                    }
                )
                
                # Stream response to client
                full_response = ""
                client_socket.sendall(b"\r\n")
                for chunk in response_stream:
                    if not chunk['done']:
                        response_part = chunk['response']
                        full_response += response_part
                        client_socket.sendall(response_part.encode('utf-8'))
                
                client_socket.sendall(b"\r\n\r\n")
                
                # Add to context manager
                client_context.add_exchange(question, full_response.strip())
                
                # Update recruiter tracker for this client
                client_recruiter_tracker.extract_info(llm_model)
                
                # Log the exchange
                log_exchange(log_file, question, full_response.strip())
                
            except Exception as e:
                error_msg = f"\r\nError generating response: {e}\r\n"
                client_socket.sendall(error_msg.encode('utf-8'))
                
    except Exception as e:
        print(f"Error handling client {client_address}: {e}")
    finally:
        client_socket.close()
        print(f"Client {client_address} disconnected")


def run_telnet_server(port, llm_model, embedding_model, system_prompt,
                     text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                     context_manager, log_file, top_k_chunks, temperature, top_p, first_name):
    """
    Run a telnet server that accepts connections and handles chatbot interactions.
    
    Args:
        port: Port number to listen on
        llm_model: LLM model name
        embedding_model: Embedding model name
        system_prompt: System prompt for the chatbot
        text_chunks: List of text chunks
        all_embeddings: Numpy array of embeddings
        personal_info_chunk_indices: List of indices for personal info chunks
        extracted_name: Extracted name from resume
        context_manager: ContextManager instance (used as template for per-client contexts)
        log_file: Log file handle
        top_k_chunks: Number of top chunks to retrieve
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        first_name: First name for display
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Allow periodic checking for interrupts
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🌐 Telnet Server Started{Colors.RESET}")
        print(f"{Colors.DIM}Listening on port {port}...{Colors.RESET}")
        print(f"{Colors.DIM}Connect with: telnet localhost {port}{Colors.RESET}")
        print(f"{Colors.DIM}Press Ctrl+C to stop the server{Colors.RESET}\n")
        
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                client_socket.settimeout(30.0)  # Timeout for client operations
                print(f"Client connected from {client_address[0]}:{client_address[1]}")
                
                # Handle client in a new thread
                client_thread = threading.Thread(
                    target=handle_telnet_client,
                    args=(client_socket, client_address, llm_model, embedding_model, system_prompt,
                          text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                          context_manager, log_file, top_k_chunks, temperature, top_p, first_name),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                # Timeout is expected, continue listening
                continue
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Shutting down telnet server...{Colors.RESET}")
                break
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue
                
    except Exception as e:
        print(f"Error starting telnet server: {e}")
    finally:
        server_socket.close()


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
    parser.add_argument(
        "-s",
        "--sms",
        action="store_true",
        help="Enable SMS/RCS messaging mode with typing indicators."
    )
    parser.add_argument(
        "-t",
        "--telnet",
        action="store_true",
        help="Enable telnet server mode."
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=2323,
        help="Port number for telnet server (default: 2323)."
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
    
    # Display startup message based on mode
    if args.sms:
        print(f"\n{Colors.BOLD}{Colors.GREEN}📱 SMS Mode Enabled{Colors.RESET}")
        print(f"{Colors.DIM}Ready to chat! (Setup took {end_time - start_time:.2f}s){Colors.RESET}")
        print(f"{Colors.GRAY}Type your message and press Enter. Say 'bye' or 'quit' to exit.{Colors.RESET}\n")
    else:
        print(f"\n--- Ready to chat! (Setup took {end_time - start_time:.2f}s) ---")
        print("Ask any question about the resume. Type 'quit' to exit.")
        print(f"Context management: Max history tokens={max_history_tokens}, Min recent messages={min_recent_messages}")

    # --- Initialize logging ---
    log_file, log_filename, log_filepath = initialize_logging(logs_dir)
    if not args.sms:
        print(f"Conversation log: {log_filename}")
    else:
        print(f"{Colors.DIM}Conversation log: {log_filename}{Colors.RESET}")
    
    # --- Create recruiter info tracker ---
    recruiter_tracker = RecruiterInfoTracker(applicant_name=extracted_name)
    
    # --- Create context manager to handle conversation history ---
    context_manager = ContextManager(
        max_history_tokens=max_history_tokens,
        min_recent_messages=min_recent_messages,
        summary_threshold=summary_threshold,
        llm_model=llm_model
    )
    
    # --- Create typing indicator for SMS mode ---
    typing_indicator = TypingIndicator() if args.sms else None
    
    # --- Extract first name for SMS display ---
    first_name = get_first_name(extracted_name) if extracted_name else None

    # 2. Start the appropriate interface
    if args.telnet:
        # Run telnet server
        try:
            run_telnet_server(
                args.port, llm_model, embedding_model, system_prompt,
                text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                context_manager, log_file, top_k_chunks, temperature, top_p, first_name
            )
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        finally:
            # Close log file and update header
            try:
                recruiter_tracker.extract_info(llm_model)
                if recruiter_tracker.has_any_info():
                    update_log_header(log_filepath, recruiter_tracker)
                
                session_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"{'='*80}\n")
                log_file.write(f"Telnet Server Session Ended: {session_end}\n")
                log_file.write(f"{'='*80}\n")
                log_file.close()
            except:
                pass  # Ignore errors when closing log file
        return
    
    # 2. Start the chat loop (normal or SMS mode)
    try:
        while True:
            if args.sms:
                question = input(f"{Colors.BLUE}{Colors.BOLD}You: {Colors.RESET}")
            else:
                question = input("\n> ")
            
            # Check for quit intent
            if detect_quit_intent(question):
                # Generate a natural goodbye response
                history_string = context_manager.get_formatted_history()
                
                if args.sms:
                    # Show typing indicator for goodbye (short delay for reading)
                    goodbye_reading_delay = random.uniform(0.3, 0.8)
                    typing_indicator.start(initial_delay=goodbye_reading_delay)
                
                goodbye_message = generate_goodbye(
                    llm_model, system_prompt, history_string, extracted_name, temperature, top_p
                )
                
                if args.sms:
                    display_sms_response(goodbye_message, typing_indicator, sender_name=first_name)
                else:
                    print(f"\n{goodbye_message}")
                
                # Log the goodbye exchange
                log_exchange(log_file, question, goodbye_message)
                break
            
            if not question.strip():
                continue

            if not args.sms:
                print("\nThinking...")
            elif args.sms:
                # Calculate reading delay based on message length
                # Base delay: 0.3 seconds, plus 0.01 seconds per character (capped at reasonable max)
                message_length = len(question)
                reading_delay = 0.3 + (message_length * 0.01)
                # Cap the delay between 0.3 and 2.0 seconds
                reading_delay = min(max(reading_delay, 0.3), 2.0)
                # Add some randomness (±20%)
                reading_delay = reading_delay * random.uniform(0.8, 1.2)
                
                # Start typing indicator with reading delay
                typing_indicator.start(initial_delay=reading_delay)
            
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
                if args.sms:
                    # In SMS mode, collect the full response first, then display it
                    for chunk in response_stream:
                        if not chunk['done']:
                            response_part = chunk['response']
                            full_response += response_part
                    # Display with typing simulation
                    display_sms_response(full_response.strip(), typing_indicator, sender_name=first_name)
                else:
                    # In normal mode, stream the response as it's generated
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
                
                # Optional: Show context stats (only in normal mode, not SMS)
                if not args.sms:
                    history_tokens = context_manager.get_history_token_count()
                    if history_tokens > max_history_tokens * 0.7:
                        print(f"\n[Context: {history_tokens}/{max_history_tokens} tokens used]")

            except Exception as e:
                # Stop typing indicator if it's running
                if args.sms and typing_indicator:
                    typing_indicator.stop()
                
                error_msg = f"An error occurred while generating the response: {e}"
                if args.sms:
                    print(f"\n{Colors.YELLOW}⚠ {error_msg}{Colors.RESET}")
                else:
                    print(f"\n{error_msg}")
                
                # Log the error
                try:
                    error_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"[{error_timestamp}] Error: {str(e)}\n\n")
                    log_file.flush()
                except:
                    pass  # Ignore logging errors

    except KeyboardInterrupt:
        if args.sms:
            if typing_indicator:
                typing_indicator.stop()
            print(f"\n{Colors.GRAY}Goodbye!{Colors.RESET}")
        else:
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

