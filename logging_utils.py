"""
Logging utilities for conversation logging and log file management.
"""
import os
import re
from datetime import datetime


def initialize_logging(logs_dir):
    """
    Creates the logs directory and opens a timestamped log file for this session.
    Creates/updates a 'latest.log' symlink pointing to the current log file.
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
    
    # Set restrictive file permissions (owner read/write only)
    try:
        os.chmod(log_filepath, 0o600)
    except Exception:
        # Silently fail if chmod fails (e.g., on Windows)
        pass
    
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
    
    # Create/update symlink to latest log
    latest_log_link = os.path.join(logs_dir, 'latest.log')
    try:
        # Remove existing symlink if it exists
        if os.path.exists(latest_log_link) or os.path.islink(latest_log_link):
            os.remove(latest_log_link)
        # Create new symlink pointing to current log file
        os.symlink(log_filename, latest_log_link)
    except Exception as e:
        # Silently fail if symlink creation fails (e.g., on Windows without admin rights)
        pass
    
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

