"""
Logging utilities for session logs and main system log.
"""
import os
import sys
from datetime import datetime


def ensure_log_directory(logs_dir):
    """
    Ensure the log directory exists, creating it if necessary.
    
    Args:
        logs_dir: Path to the log directory
    
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
            # Set restrictive permissions (700) for the directory
            os.chmod(logs_dir, 0o700)
        return True
    except OSError as e:
        print(f"Error creating log directory '{logs_dir}': {e}", file=sys.stderr)
        return False


def get_main_log_path(logs_dir):
    """
    Get the path to the main system log file.
    
    Args:
        logs_dir: Path to the log directory
    
    Returns:
        str: Path to the main log file
    """
    return os.path.join(logs_dir, 'cv-caddy.log')


def get_session_log_path(logs_dir, session_id=None):
    """
    Get the path to a session log file.
    
    Args:
        logs_dir: Path to the log directory
        session_id: Optional session identifier (defaults to timestamp)
    
    Returns:
        str: Path to the session log file
    """
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(logs_dir, f"session_{session_id}.log")


def write_main_log(logs_dir, level, message):
    """
    Write a message to the main system log file.
    
    Args:
        logs_dir: Path to the log directory
        level: Log level (INFO, WARNING, ERROR, etc.)
        message: Message to log
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not ensure_log_directory(logs_dir):
        return False
    
    main_log_path = get_main_log_path(logs_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(main_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
            f.flush()
        # Set restrictive permissions (600) on the log file
        os.chmod(main_log_path, 0o600)
        return True
    except OSError as e:
        print(f"Error writing to main log '{main_log_path}': {e}", file=sys.stderr)
        return False


def initialize_session_log(logs_dir, session_type="interactive", ip_address=None):
    """
    Initialize a new session log file.
    
    Args:
        logs_dir: Path to the log directory
        session_type: Type of session (interactive, telnet, etc.)
        ip_address: Optional IP address of the client (for telnet sessions)
    
    Returns:
        tuple: (log_file_handle, log_filepath, session_id) or (None, None, None) on error
    """
    if not ensure_log_directory(logs_dir):
        return None, None, None
    
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = get_session_log_path(logs_dir, session_id)
    
    try:
        log_file = open(log_filepath, 'w', encoding='utf-8')
        session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_file.write(f"{'='*80}\n")
        if ip_address:
            log_file.write(f"Inbound IP: {ip_address}\n")
        log_file.write(f"Session Started: {session_start}\n")
        log_file.write(f"Session Type: {session_type}\n")
        log_file.write(f"Session ID: {session_id}\n")
        log_file.write(f"{'='*80}\n\n")
        log_file.flush()
        
        # Set restrictive permissions (600) on the log file
        os.chmod(log_filepath, 0o600)
        
        # Log session start to main log
        write_main_log(logs_dir, "INFO", f"Session started: {session_type} (ID: {session_id})")
        
        return log_file, log_filepath, session_id
    except OSError as e:
        print(f"Error creating session log '{log_filepath}': {e}", file=sys.stderr)
        write_main_log(logs_dir, "ERROR", f"Failed to create session log: {e}")
        return None, None, None


def log_session_exchange(log_file, user_message, bot_response):
    """
    Log a conversation exchange to a session log file.
    
    Args:
        log_file: Open file handle for the session log
        user_message: User's message
        bot_response: Bot's response
    """
    if log_file is None:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] User:\n{user_message}\n\n")
        log_file.write(f"[{timestamp}] Bot:\n{bot_response}\n\n")
        log_file.write(f"{'-'*80}\n\n")
        log_file.flush()
    except (OSError, AttributeError):
        pass  # Ignore errors when logging


def close_session_log(log_file, log_filepath, logs_dir, session_id, interrupted=False):
    """
    Close a session log file and log the session end.
    
    Args:
        log_file: Open file handle for the session log
        log_filepath: Path to the session log file
        logs_dir: Path to the log directory
        session_id: Session identifier
        interrupted: True if session was interrupted, False otherwise
    """
    if log_file is None:
        return
    
    try:
        session_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Interrupted" if interrupted else "Ended"
        log_file.write(f"{'='*80}\n")
        log_file.write(f"Session {status}: {session_end}\n")
        log_file.write(f"{'='*80}\n")
        log_file.close()
        
        # Log session end to main log
        write_main_log(logs_dir, "INFO", f"Session {status.lower()}: {session_id}")
    except (OSError, AttributeError):
        pass  # Ignore errors when closing


def log_system_event(logs_dir, level, message):
    """
    Log a system event to the main log file.
    Convenience wrapper around write_main_log.
    
    Args:
        logs_dir: Path to the log directory
        level: Log level (INFO, WARNING, ERROR, etc.)
        message: Message to log
    """
    write_main_log(logs_dir, level, message)

