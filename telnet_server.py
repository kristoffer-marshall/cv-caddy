"""
Telnet server implementation for CV Caddy.
Handles telnet connections, NAWS negotiation, and client interactions.
"""
import socket
import time
import threading
import ollama

from context_manager import ContextManager
from recruiter_tracker import RecruiterInfoTracker
from logging_utils import (
    initialize_session_log, log_session_exchange, close_session_log,
    log_system_event
)
from security import (
    validate_input_length, sanitize_input, detect_prompt_injection,
    sanitize_error_message
)
from daemon_utils import (
    check_shutdown_requested, get_default_status_file,
    update_status_file, systemd_notify
)
from rag import find_relevant_chunks


# Telnet protocol constants
IAC = 255  # Interpret As Command
WILL = 251
WONT = 252
DO = 253
DONT = 254
SB = 250  # Subnegotiation Begin
SE = 240  # Subnegotiation End
NAWS = 31  # Negotiate About Window Size


def negotiate_telnet_options(client_socket, default_width=120, logs_dir=None):
    """
    Negotiate telnet options, specifically NAWS (Negotiate About Window Size).
    
    Args:
        client_socket: The socket connection to the client
        default_width: Default width to use if NAWS negotiation fails
        logs_dir: Path to log directory for logging (optional)
    
    Returns:
        int: Terminal width from NAWS, or default_width if not available
    """
    terminal_width = default_width
    
    try:
        # Request NAWS from client
        client_socket.sendall(bytes([IAC, DO, NAWS]))
        
        # Set a short timeout for negotiation
        original_timeout = client_socket.gettimeout()
        client_socket.settimeout(2.0)
        
        try:
            # Read and process telnet option negotiation
            buffer = b''
            max_negotiation_bytes = 1024
            bytes_read = 0
            
            while bytes_read < max_negotiation_bytes:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                buffer += data
                bytes_read += len(data)
                
                # Look for NAWS subnegotiation: IAC SB NAWS width_high width_low height_high height_low IAC SE
                iac_pos = buffer.find(IAC)
                while iac_pos != -1 and iac_pos < len(buffer) - 1:
                    if iac_pos + 2 < len(buffer):
                        cmd = buffer[iac_pos + 1]
                        
                        # Check for subnegotiation (SB)
                        if cmd == SB:
                            # Look for SE to find end of subnegotiation
                            se_pos = buffer.find(bytes([IAC, SE]), iac_pos + 2)
                            if se_pos != -1:
                                subneg_data = buffer[iac_pos + 2:se_pos]
                                if len(subneg_data) >= 1 and subneg_data[0] == NAWS:
                                    # Parse NAWS data: width_high, width_low, height_high, height_low
                                    if len(subneg_data) >= 5:
                                        width_high = subneg_data[1]
                                        width_low = subneg_data[2]
                                        height_high = subneg_data[3]
                                        height_low = subneg_data[4]
                                        
                                        # Calculate width (16-bit value)
                                        width = (width_high << 8) | width_low
                                        height = (height_high << 8) | height_low
                                        
                                        if width > 0 and width < 1000:  # Sanity check
                                            terminal_width = width
                                            if logs_dir:
                                                log_system_event(logs_dir, "INFO", 
                                                    f"NAWS negotiation successful: {width}x{height}")
                                        
                                        # Remove processed subnegotiation from buffer
                                        buffer = buffer[se_pos + 2:]
                                        iac_pos = buffer.find(IAC)
                                        continue
                        
                        # Handle WILL/WONT responses for NAWS
                        elif cmd in (WILL, WONT) and iac_pos + 2 < len(buffer):
                            option = buffer[iac_pos + 2]
                            if option == NAWS:
                                # Client responded to our DO NAWS
                                if cmd == WILL:
                                    # Client will send window size, wait for it
                                    pass
                                else:
                                    # Client won't send window size, use default
                                    break
                                buffer = buffer[iac_pos + 3:]
                                iac_pos = buffer.find(IAC)
                                continue
                    
                    # Move past this IAC
                    iac_pos = buffer.find(IAC, iac_pos + 1)
                
                # If we've processed all IACs and have a valid width, we're done
                if terminal_width != default_width:
                    break
                
                # Small delay to allow more data to arrive
                time.sleep(0.1)
                
        except socket.timeout:
            # Timeout is OK, use default width
            pass
        except Exception as e:
            # Log but don't fail - use default width
            if logs_dir:
                log_system_event(logs_dir, "WARNING", f"Error during NAWS negotiation: {e}")
        finally:
            # Restore original timeout
            client_socket.settimeout(original_timeout)
            
    except Exception as e:
        # If negotiation fails, just use default width
        if logs_dir:
            log_system_event(logs_dir, "WARNING", f"NAWS negotiation failed: {e}")
    
    return terminal_width


def handle_telnet_client(client_socket, client_address, llm_model, embedding_model, system_prompt,
                         text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                         context_manager, top_k_chunks, temperature, top_p, first_name,
                         initial_greeting_template, rate_limiter, thinking_message, banner="",
                         logs_dir=None, telnet_line_width=120, wrap_text_func=None,
                         generate_initial_greeting_func=None, generate_goodbye_func=None,
                         detect_quit_intent_func=None, StreamingTextWrapper_class=None, Colors_class=None):
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
        top_k_chunks: Number of top chunks to retrieve
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        first_name: First name for display
        initial_greeting_template: Template for initial greeting (may contain ${NAME})
        rate_limiter: RateLimiter instance for rate limiting
        thinking_message: Message to display while processing
        banner: Banner message to display at conversation start
        logs_dir: Path to log directory for session logging
        telnet_line_width: Line width for word wrapping in telnet connections (default: 120)
        wrap_text_func: Function to wrap text (from main.py)
        generate_initial_greeting_func: Function to generate initial greeting (from main.py)
        generate_goodbye_func: Function to generate goodbye (from main.py)
        detect_quit_intent_func: Function to detect quit intent (from main.py)
        StreamingTextWrapper_class: StreamingTextWrapper class (from main.py)
        Colors_class: Colors class (from main.py)
    """
    ip_address = client_address[0]
    
    # Initialize session log for this client
    session_log_file = None
    session_log_filepath = None
    session_id = None
    if logs_dir:
        session_log_file, session_log_filepath, session_id = initialize_session_log(
            logs_dir, session_type=f"telnet_{ip_address}", ip_address=ip_address
        )
        if session_log_file:
            log_system_event(logs_dir, "INFO", f"Telnet client connected from {ip_address}:{client_address[1]} (Session: {session_id})")
    
    try:
        # Create a new context manager for this client
        client_context = ContextManager(
            max_history_tokens=context_manager.max_history_tokens,
            min_recent_messages=context_manager.min_recent_messages,
            summary_threshold=context_manager.summary_threshold,
            llm_model=context_manager.llm_model
        )
        
        # Create a recruiter tracker for this client
        client_recruiter_tracker = RecruiterInfoTracker(applicant_name=extracted_name)
        
        # Negotiate telnet options (NAWS for terminal size)
        actual_terminal_width = negotiate_telnet_options(client_socket, telnet_line_width, logs_dir)
        
        # Send banner if configured
        if banner and banner.strip():
            client_socket.sendall(f"\r\n{banner}\r\n\r\n".encode('utf-8'))
        
        # Generate and send initial greeting
        initial_greeting = generate_initial_greeting_func(
            llm_model, system_prompt, initial_greeting_template, extracted_name,
            text_chunks, all_embeddings, personal_info_chunk_indices,
            top_k_chunks, embedding_model, temperature, top_p
        )
        # Wrap greeting text before sending (use negotiated width)
        wrapped_greeting = wrap_text_func(initial_greeting, width=actual_terminal_width)
        client_socket.sendall(f"\r\n{wrapped_greeting}\r\n\r\n".encode('utf-8'))
        
        # Add initial greeting to context manager so it's remembered
        # We add it as a Bot message with an empty User message to represent the start
        client_context.history.append(("Bot", initial_greeting))
        
        # Log initial greeting
        if session_log_file:
            log_session_exchange(session_log_file, "[Conversation started]", initial_greeting)
        
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
            
            # Check rate limit
            if rate_limiter:
                allowed, remaining = rate_limiter.check_rate_limit(ip_address)
                if not allowed:
                    client_socket.sendall(b"\r\nError: Rate limit exceeded. Please try again later.\r\n")
                    continue
            
            # Validate and sanitize input
            is_valid, length_error = validate_input_length(question)
            if not is_valid:
                client_socket.sendall(f"\r\nError: {length_error}\r\n".encode('utf-8'))
                continue
            
            # Sanitize input
            question = sanitize_input(question)
            
            # Check for prompt injection
            is_suspicious, suspicious_patterns = detect_prompt_injection(question)
            if is_suspicious:
                # Log the attempt but still process (with mitigation in prompt)
                print(f"Warning: Suspicious input detected from {client_address[0]}: {suspicious_patterns}")
                # Continue processing but will add mitigation in prompt construction
            
            # Check for quit intent
            if detect_quit_intent_func(question):
                history_string = client_context.get_formatted_history()
                goodbye_message = generate_goodbye_func(
                    llm_model, system_prompt, history_string, extracted_name, temperature, top_p
                )
                # Wrap goodbye message before sending (use negotiated width)
                wrapped_goodbye = wrap_text_func(goodbye_message, width=actual_terminal_width)
                client_socket.sendall(f"\r\n{wrapped_goodbye}\r\n".encode('utf-8'))
                
                # Log goodbye exchange
                if session_log_file:
                    log_session_exchange(session_log_file, question, goodbye_message)
                
                break
            
            # Get embedding for the question
            try:
                query_response = ollama.embeddings(
                    model=embedding_model,
                    prompt=question
                )
                query_embedding = query_response["embedding"]
            except Exception as e:
                error_msg = sanitize_error_message(e, include_details=False)
                client_socket.sendall(f"\r\n{error_msg}\r\n".encode('utf-8'))
                # Log detailed error server-side
                detailed_error = sanitize_error_message(e, include_details=True)
                print(f"Error getting embedding for query from {client_address[0]}: {detailed_error}")
                if logs_dir:
                    log_system_event(logs_dir, "ERROR", f"Error getting embedding for query from {ip_address}: {detailed_error}")
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
                
                # Stream response to client with word wrapping
                full_response = ""
                client_socket.sendall(b"\r\n")
                
                # Create a wrapper that sends to socket
                def socket_output(text, **kwargs):
                    client_socket.sendall(text.encode('utf-8'))
                
                stream_wrapper = StreamingTextWrapper_class(width=actual_terminal_width, output_func=socket_output)
                for chunk in response_stream:
                    if not chunk['done']:
                        response_part = chunk['response']
                        stream_wrapper.write(response_part)
                        full_response += response_part
                stream_wrapper.finish()  # Flush and add final newline
                
                client_socket.sendall(b"\r\n")
                
                # Add to context manager
                client_context.add_exchange(question, full_response.strip())
                
                # Update recruiter tracker for this client
                client_recruiter_tracker.extract_info(llm_model)
                
                # Log the exchange
                if session_log_file:
                    log_session_exchange(session_log_file, question, full_response.strip())
                
            except Exception as e:
                error_msg = sanitize_error_message(e, include_details=False)
                client_socket.sendall(f"\r\n{error_msg}\r\n".encode('utf-8'))
                # Log detailed error server-side
                detailed_error = sanitize_error_message(e, include_details=True)
                print(f"Error generating response for {client_address[0]}: {detailed_error}")
                if logs_dir:
                    log_system_event(logs_dir, "ERROR", f"Error generating response for {ip_address}: {detailed_error}")
                
    except Exception as e:
        error_msg = sanitize_error_message(e, include_details=True)
        print(f"Error handling client {client_address}: {error_msg}")
        if logs_dir:
            log_system_event(logs_dir, "ERROR", f"Error handling telnet client {ip_address}: {error_msg}")
    finally:
        # Close session log
        if session_log_file:
            close_session_log(session_log_file, session_log_filepath, logs_dir, session_id, interrupted=False)
        
        if logs_dir:
            log_system_event(logs_dir, "INFO", f"Telnet client disconnected: {ip_address}:{client_address[1]}")
        
        # Decrement connection count
        if rate_limiter:
            rate_limiter.decrement_connections(ip_address)
        client_socket.close()
        print(f"Client {client_address} disconnected")


def run_telnet_server(port, llm_model, embedding_model, system_prompt,
                     text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                     context_manager, top_k_chunks, temperature, top_p, first_name,
                     initial_greeting_template, rate_limiter, bind_address='127.0.0.1', 
                     connection_timeout=30.0, idle_timeout=300.0, shutdown_message=None,
                     active_connections=None, server_socket_ref=None, thinking_message="Thinking...",
                     banner="", logs_dir=None, status_file_path=None, telnet_line_width=120,
                     wrap_text_func=None, generate_initial_greeting_func=None,
                     generate_goodbye_func=None, detect_quit_intent_func=None,
                     StreamingTextWrapper_class=None, Colors_class=None):
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
        top_k_chunks: Number of top chunks to retrieve
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        first_name: First name for display
        initial_greeting_template: Template for initial greeting (may contain ${NAME})
        rate_limiter: RateLimiter instance for rate limiting
        bind_address: Address to bind to (default: 127.0.0.1 for localhost only)
        connection_timeout: Timeout for client operations in seconds (default: 30.0)
        idle_timeout: Idle timeout in seconds (default: 300.0)
        shutdown_message: Message to send to active connections on shutdown
        active_connections: List to track active connections for graceful shutdown
        server_socket_ref: Reference to server socket for reload/shutdown handling
        thinking_message: Message to display while processing
        banner: Banner message to display at conversation start
        logs_dir: Path to log directory for logging
        status_file_path: Path to status file for health monitoring
        telnet_line_width: Line width for word wrapping in telnet connections (default: 120)
        wrap_text_func: Function to wrap text (from main.py)
        generate_initial_greeting_func: Function to generate initial greeting (from main.py)
        generate_goodbye_func: Function to generate goodbye (from main.py)
        detect_quit_intent_func: Function to detect quit intent (from main.py)
        StreamingTextWrapper_class: StreamingTextWrapper class (from main.py)
        Colors_class: Colors class (from main.py)
    """
    if logs_dir:
        log_system_event(logs_dir, "INFO", f"Starting telnet server on {bind_address}:{port}")
    
    server_socket = None
    last_connection_time = time.time()
    last_watchdog_update = time.time()
    watchdog_interval = 30.0  # Update watchdog every 30 seconds
    connection_count = 0
    error_count = 0
    
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Store socket reference if provided (for reload/shutdown handling)
        if server_socket_ref is not None:
            server_socket_ref[0] = server_socket
        
        if active_connections is None:
            active_connections = []
        
        # Warn if binding to all interfaces
        if bind_address == '0.0.0.0':
            warning_msg = "Binding to 0.0.0.0 exposes the server to all network interfaces. Consider using 127.0.0.1 for localhost-only access."
            print(f"{Colors_class.YELLOW}Warning: {warning_msg}{Colors_class.RESET}")
            print(f"{Colors_class.YELLOW}Consider using 127.0.0.1 for localhost-only access.{Colors_class.RESET}\n")
            if logs_dir:
                log_system_event(logs_dir, "WARNING", warning_msg)
        
        server_socket.bind((bind_address, port))
        server_socket.listen(5)
        server_socket.settimeout(1.0)  # Allow periodic checking for interrupts
        
        # Update status: server is listening
        if status_file_path:
            update_status_file(status_file_path, {
                'status': 'running',
                'state': 'listening',
                'bind_address': bind_address,
                'port': port,
                'active_connections': len(active_connections),
                'total_connections': connection_count,
                'error_count': error_count,
                'last_connection_time': last_connection_time
            })
        
        # Notify systemd that we're ready
        systemd_notify(ready=True, status=f"Listening on {bind_address}:{port}")
        
        print(f"\n{Colors_class.BOLD}{Colors_class.GREEN}🌐 Telnet Server Started{Colors_class.RESET}")
        if logs_dir:
            log_system_event(logs_dir, "INFO", f"Telnet server started successfully on {bind_address}:{port}")
        print(f"{Colors_class.DIM}Listening on {bind_address}:{port}...{Colors_class.RESET}")
        if bind_address == '127.0.0.1':
            print(f"{Colors_class.DIM}Connect with: telnet localhost {port}{Colors_class.RESET}")
        else:
            print(f"{Colors_class.DIM}Connect with: telnet {bind_address} {port}{Colors_class.RESET}")
        print(f"{Colors_class.DIM}Press Ctrl+C to stop the server{Colors_class.RESET}\n")
        
        while not check_shutdown_requested():
            try:
                # Update watchdog periodically
                current_time = time.time()
                if current_time - last_watchdog_update >= watchdog_interval:
                    # Check if server socket is still valid
                    try:
                        # Try to get socket state
                        server_socket.getsockname()
                        socket_valid = True
                    except (OSError, AttributeError):
                        socket_valid = False
                        error_msg = "Server socket is no longer valid"
                        print(f"ERROR: {error_msg}")
                        if logs_dir:
                            log_system_event(logs_dir, "ERROR", error_msg)
                        if status_file_path:
                            update_status_file(status_file_path, {
                                'status': 'error',
                                'state': 'socket_invalid',
                                'error': error_msg,
                                'bind_address': bind_address,
                                'port': port,
                                'active_connections': len(active_connections),
                                'total_connections': connection_count,
                                'error_count': error_count,
                                'last_connection_time': last_connection_time
                            })
                        systemd_notify(status=f"ERROR: {error_msg}")
                        break  # Exit loop to restart server
                    
                    # Update status file
                    if status_file_path:
                        update_status_file(status_file_path, {
                            'status': 'running',
                            'state': 'listening',
                            'bind_address': bind_address,
                            'port': port,
                            'active_connections': len(active_connections),
                            'total_connections': connection_count,
                            'error_count': error_count,
                            'last_connection_time': last_connection_time,
                            'socket_valid': socket_valid,
                            'uptime_seconds': current_time - last_connection_time if last_connection_time else 0
                        })
                    
                    # Update systemd watchdog
                    systemd_notify(watchdog=True, status=f"Listening on {bind_address}:{port}, {len(active_connections)} active connections")
                    last_watchdog_update = current_time
                
                client_socket, client_address = server_socket.accept()
                client_socket.settimeout(connection_timeout)
                ip_address = client_address[0]
                connection_count += 1
                last_connection_time = time.time()
                
                # Check connection limit per IP
                if rate_limiter:
                    if not rate_limiter.increment_connections(ip_address):
                        error_msg = f"Connection limit exceeded for {ip_address}, rejecting connection"
                        print(error_msg)
                        if logs_dir:
                            log_system_event(logs_dir, "WARNING", error_msg)
                        try:
                            client_socket.sendall(b"Error: Connection limit exceeded. Please try again later.\r\n")
                        except Exception:
                            pass
                        try:
                            client_socket.close()
                        except Exception:
                            pass
                        continue
                
                print(f"Client connected from {ip_address}:{client_address[1]} (total: {connection_count})")
                if logs_dir:
                    log_system_event(logs_dir, "INFO", f"Client connected from {ip_address}:{client_address[1]}")
                
                # Track connection for graceful shutdown
                active_connections.append(client_socket)
                
                # Update status with new connection
                if status_file_path:
                    update_status_file(status_file_path, {
                        'status': 'running',
                        'state': 'listening',
                        'bind_address': bind_address,
                        'port': port,
                        'active_connections': len(active_connections),
                        'total_connections': connection_count,
                        'error_count': error_count,
                        'last_connection_time': last_connection_time
                    })
                
                # Handle client in a new thread
                def client_handler_wrapper(sock, addr):
                    try:
                        handle_telnet_client(
                            sock, addr, llm_model, embedding_model, system_prompt,
                            text_chunks, all_embeddings, personal_info_chunk_indices, extracted_name,
                            context_manager, top_k_chunks, temperature, top_p, first_name,
                            initial_greeting_template, rate_limiter, thinking_message, banner,
                            logs_dir, telnet_line_width, wrap_text_func,
                            generate_initial_greeting_func, generate_goodbye_func,
                            detect_quit_intent_func, StreamingTextWrapper_class, Colors_class
                        )
                    except Exception as e:
                        error_msg = f"Error in client handler for {addr}: {e}"
                        print(f"ERROR: {error_msg}")
                        if logs_dir:
                            log_system_event(logs_dir, "ERROR", error_msg)
                        import traceback
                        if logs_dir:
                            log_system_event(logs_dir, "ERROR", f"Traceback: {traceback.format_exc()}")
                    finally:
                        if sock in active_connections:
                            active_connections.remove(sock)
                        # Update status when connection closes
                        if status_file_path:
                            update_status_file(status_file_path, {
                                'status': 'running',
                                'state': 'listening',
                                'bind_address': bind_address,
                                'port': port,
                                'active_connections': len(active_connections),
                                'total_connections': connection_count,
                                'error_count': error_count,
                                'last_connection_time': last_connection_time
                            })
                
                client_thread = threading.Thread(
                    target=client_handler_wrapper,
                    args=(client_socket, client_address),
                    daemon=True
                )
                client_thread.start()
                
            except socket.timeout:
                # Timeout is expected, continue listening
                continue
            except KeyboardInterrupt:
                print(f"\n{Colors_class.YELLOW}Shutting down telnet server...{Colors_class.RESET}")
                if logs_dir:
                    log_system_event(logs_dir, "INFO", "Shutdown requested via KeyboardInterrupt")
                break
            except OSError as e:
                if not check_shutdown_requested():
                    error_count += 1
                    error_msg = f"Socket error accepting connection: {e}"
                    print(f"ERROR: {error_msg}")
                    if logs_dir:
                        log_system_event(logs_dir, "ERROR", error_msg)
                    if status_file_path:
                        update_status_file(status_file_path, {
                            'status': 'error',
                            'state': 'socket_error',
                            'error': str(e),
                            'bind_address': bind_address,
                            'port': port,
                            'active_connections': len(active_connections),
                            'total_connections': connection_count,
                            'error_count': error_count,
                            'last_connection_time': last_connection_time
                        })
                    systemd_notify(status=f"ERROR: {error_msg}")
                    # If socket is closed, we need to exit and let the main loop restart
                    if e.errno == 9:  # EBADF - Bad file descriptor
                        print("ERROR: Server socket is closed, exiting to allow restart")
                        if logs_dir:
                            log_system_event(logs_dir, "ERROR", "Server socket closed, exiting to allow restart")
                        break
                continue
            except Exception as e:
                if not check_shutdown_requested():
                    error_count += 1
                    error_msg = f"Error accepting connection: {e}"
                    print(f"ERROR: {error_msg}")
                    if logs_dir:
                        log_system_event(logs_dir, "ERROR", error_msg)
                        import traceback
                        log_system_event(logs_dir, "ERROR", f"Traceback: {traceback.format_exc()}")
                    if status_file_path:
                        update_status_file(status_file_path, {
                            'status': 'error',
                            'state': 'accept_error',
                            'error': str(e),
                            'bind_address': bind_address,
                            'port': port,
                            'active_connections': len(active_connections),
                            'total_connections': connection_count,
                            'error_count': error_count,
                            'last_connection_time': last_connection_time
                        })
                    systemd_notify(status=f"ERROR: {error_msg}")
                continue
        
        # Graceful shutdown: notify active connections
        if shutdown_message and active_connections:
            shutdown_msg = f"Notifying {len(active_connections)} active connections of shutdown..."
            print(shutdown_msg)
            if logs_dir:
                log_system_event(logs_dir, "INFO", shutdown_msg)
            for conn in active_connections[:]:  # Copy list to avoid modification during iteration
                try:
                    conn.sendall(f"\r\n{shutdown_message}\r\n".encode('utf-8'))
                except Exception:
                    pass  # Connection may already be closed
        
        # Wait for connections to close (with timeout)
        shutdown_timeout = 10.0  # seconds
        start_time = time.time()
        while active_connections and (time.time() - start_time) < shutdown_timeout:
            time.sleep(0.5)
        
        if active_connections:
            print(f"Force closing {len(active_connections)} remaining connections...")
            if logs_dir:
                log_system_event(logs_dir, "WARNING", f"Force closing {len(active_connections)} remaining connections")
            for conn in active_connections:
                try:
                    conn.close()
                except Exception:
                    pass
        
        # Update status: server is stopping
        if status_file_path:
            update_status_file(status_file_path, {
                'status': 'stopping',
                'state': 'shutdown',
                'bind_address': bind_address,
                'port': port,
                'active_connections': 0,
                'total_connections': connection_count,
                'error_count': error_count,
                'last_connection_time': last_connection_time
            })
        
        systemd_notify(stopping=True, status="Shutting down telnet server")
                
    except Exception as e:
        error_msg = f"Error starting telnet server: {e}"
        print(f"ERROR: {error_msg}")
        if logs_dir:
            log_system_event(logs_dir, "ERROR", error_msg)
            import traceback
            log_system_event(logs_dir, "ERROR", f"Traceback: {traceback.format_exc()}")
        if status_file_path:
            update_status_file(status_file_path, {
                'status': 'error',
                'state': 'startup_error',
                'error': str(e),
                'bind_address': bind_address,
                'port': port,
                'active_connections': 0,
                'total_connections': connection_count,
                'error_count': error_count
            })
        systemd_notify(status=f"ERROR: {error_msg}")
        raise  # Re-raise to allow main loop to handle
    finally:
        if server_socket:
            try:
                server_socket.close()
            except Exception:
                pass
        print("Telnet server stopped.")
        if logs_dir:
            log_system_event(logs_dir, "INFO", "Telnet server stopped")
        if status_file_path:
            update_status_file(status_file_path, {
                'status': 'stopped',
                'state': 'stopped',
                'bind_address': bind_address,
                'port': port,
                'active_connections': 0,
                'total_connections': connection_count,
                'error_count': error_count
            })
        systemd_notify(stopping=True, status="Telnet server stopped")

