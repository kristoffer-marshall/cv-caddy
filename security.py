"""
Security utilities for input validation, sanitization, and path validation.
"""
import os
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta


# Security constants
MAX_INPUT_LENGTH = 5000
MAX_PROMPT_LENGTH = 50000  # Prevent token exhaustion attacks
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def validate_input_length(text, max_length=MAX_INPUT_LENGTH):
    """
    Validate that input text doesn't exceed maximum length.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed length (default: MAX_INPUT_LENGTH)
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text:
        return True, None
    
    if len(text) > max_length:
        return False, f"Input exceeds maximum length of {max_length} characters"
    
    return True, None


def sanitize_input(text):
    """
    Sanitize user input by removing/nullifying control characters and normalizing whitespace.
    
    Args:
        text: Input text to sanitize
    
    Returns:
        str: Sanitized text
    """
    if not text:
        return ""
    
    # Remove control characters except newline, carriage return, and tab
    # Keep: \n (0x0A), \r (0x0D), \t (0x09)
    sanitized = ""
    for char in text:
        code = ord(char)
        # Allow printable characters (32-126), newline (10), carriage return (13), tab (9)
        if (32 <= code <= 126) or code in (9, 10, 13):
            sanitized += char
        # Replace other control characters with space
        elif code < 32:
            sanitized += " "
        # Keep other unicode characters (for international text)
        else:
            sanitized += char
    
    # Normalize whitespace: collapse multiple spaces, but preserve newlines
    lines = sanitized.split('\n')
    normalized_lines = []
    for line in lines:
        # Collapse multiple spaces to single space
        normalized_line = re.sub(r' +', ' ', line.strip())
        normalized_lines.append(normalized_line)
    
    # Join lines back, preserving newlines
    result = '\n'.join(normalized_lines)
    
    # Remove excessive newlines (more than 2 consecutive)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def detect_prompt_injection(text):
    """
    Detect common prompt injection patterns in user input.
    
    Args:
        text: Input text to check
    
    Returns:
        tuple: (is_suspicious, suspicious_patterns)
    """
    if not text:
        return False, []
    
    text_lower = text.lower()
    suspicious_patterns = []
    
    # Common prompt injection patterns
    injection_patterns = [
        (r'ignore\s+(previous|above|all|earlier)\s+(instructions?|prompts?|commands?)', 'ignore_previous'),
        (r'forget\s+(previous|above|all|earlier)', 'forget_previous'),
        (r'you\s+are\s+now', 'role_override'),
        (r'act\s+as\s+(if\s+)?you\s+are', 'role_override'),
        (r'system\s*[:;]\s*', 'system_prefix'),
        (r'<\|(system|assistant|user)\|>', 'special_tokens'),
        (r'\[INST\]|\[/INST\]', 'llama_tokens'),
        (r'###\s*(system|instruction|prompt)', 'markdown_override'),
        (r'override\s+(system|instructions?)', 'override'),
        (r'new\s+(instructions?|prompt|system)', 'new_instructions'),
        (r'disregard\s+(previous|above|all)', 'disregard'),
    ]
    
    for pattern, name in injection_patterns:
        if re.search(pattern, text_lower):
            suspicious_patterns.append(name)
    
    # Check for excessive repetition (potential DoS)
    if len(text) > 100:
        # Check for repeated phrases
        words = text_lower.split()
        if len(words) > 20:
            word_counts = defaultdict(int)
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_counts[word] += 1
            # If any word appears more than 30% of the time, it's suspicious
            max_count = max(word_counts.values()) if word_counts else 0
            if max_count > len(words) * 0.3:
                suspicious_patterns.append('excessive_repetition')
    
    is_suspicious = len(suspicious_patterns) > 0
    return is_suspicious, suspicious_patterns


def validate_file_path(file_path, base_dir=None, allow_absolute=False):
    """
    Validate that a file path doesn't contain path traversal and stays within allowed directory.
    
    Args:
        file_path: Path to validate
        base_dir: Base directory to restrict paths to (default: PROJECT_ROOT)
        allow_absolute: Whether to allow absolute paths (default: False)
    
    Returns:
        tuple: (is_valid, normalized_path, error_message)
    """
    if not file_path:
        return False, None, "Empty file path"
    
    # Check for path traversal attempts
    if '..' in file_path:
        return False, None, "Path traversal detected (..)"
    
    # Check for null bytes
    if '\x00' in file_path:
        return False, None, "Null byte detected in path"
    
    # Normalize the path
    try:
        normalized = os.path.normpath(file_path)
    except Exception as e:
        return False, None, f"Invalid path format: {str(e)}"
    
    # If absolute path is not allowed, check if it's absolute
    if not allow_absolute and os.path.isabs(normalized):
        return False, None, "Absolute paths not allowed"
    
    # If base_dir is specified, ensure path is within it
    if base_dir:
        base_dir = os.path.normpath(base_dir)
        # Resolve to absolute paths for comparison
        abs_base = os.path.abspath(base_dir)
        abs_path = os.path.abspath(os.path.join(base_dir, normalized))
        
        # Check if resolved path is within base directory
        try:
            common_path = os.path.commonpath([abs_base, abs_path])
            if common_path != abs_base:
                return False, None, "Path outside allowed directory"
        except ValueError:
            # Paths on different drives (Windows) or invalid
            return False, None, "Path outside allowed directory"
        
        return True, normalized, None
    
    return True, normalized, None


def validate_and_normalize_path(file_path, base_dir=None):
    """
    Validate and normalize a file path, ensuring it's safe to use.
    
    Args:
        file_path: Path to validate and normalize
        base_dir: Base directory (default: PROJECT_ROOT)
    
    Returns:
        str: Normalized, safe path, or None if invalid
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT
    
    is_valid, normalized, error = validate_file_path(file_path, base_dir, allow_absolute=False)
    
    if not is_valid:
        return None
    
    # Join with base directory to get full path
    full_path = os.path.join(base_dir, normalized)
    return os.path.normpath(full_path)


class RateLimiter:
    """
    Simple rate limiter for IP addresses.
    """
    
    def __init__(self, max_requests=60, window_seconds=60, max_connections=10):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window (default: 60)
            window_seconds: Time window in seconds (default: 60)
            max_connections: Maximum concurrent connections per IP (default: 10)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_connections = max_connections
        self.requests = defaultdict(list)  # IP -> list of request timestamps
        self.connections = defaultdict(int)  # IP -> current connection count
        self.lock = threading.Lock()
    
    def check_rate_limit(self, ip_address):
        """
        Check if IP address has exceeded rate limit.
        
        Args:
            ip_address: IP address to check
        
        Returns:
            tuple: (allowed, remaining_requests)
        """
        with self.lock:
            now = datetime.now()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests outside the window
            if ip_address in self.requests:
                self.requests[ip_address] = [
                    ts for ts in self.requests[ip_address]
                    if ts > window_start
                ]
            
            # Check if limit exceeded
            request_count = len(self.requests.get(ip_address, []))
            if request_count >= self.max_requests:
                return False, 0
            
            # Record this request
            self.requests[ip_address].append(now)
            remaining = self.max_requests - request_count - 1
            
            return True, remaining
    
    def increment_connections(self, ip_address):
        """
        Increment connection count for IP address.
        
        Args:
            ip_address: IP address
        
        Returns:
            bool: True if connection allowed, False if limit exceeded
        """
        with self.lock:
            if self.connections[ip_address] >= self.max_connections:
                return False
            self.connections[ip_address] += 1
            return True
    
    def decrement_connections(self, ip_address):
        """Decrement connection count for IP address."""
        with self.lock:
            if ip_address in self.connections:
                self.connections[ip_address] = max(0, self.connections[ip_address] - 1)
                # Clean up if zero
                if self.connections[ip_address] == 0:
                    del self.connections[ip_address]


# Global rate limiter instance (will be initialized in main.py)
rate_limiter = None


def sanitize_error_message(error, include_details=False):
    """
    Sanitize error messages to prevent information disclosure.
    
    Args:
        error: Exception or error message
        include_details: Whether to include detailed error (for server-side logging)
    
    Returns:
        str: Sanitized error message
    """
    if isinstance(error, Exception):
        error_str = str(error)
        error_type = type(error).__name__
    else:
        error_str = str(error)
        error_type = "Error"
    
    # Generic client-facing message
    client_message = "An error occurred while processing your request. Please try again."
    
    if include_details:
        # Server-side: include more details but still sanitize
        # Remove file paths, system info
        sanitized = error_str
        # Remove absolute paths
        sanitized = re.sub(r'/[^\s]+', '[path]', sanitized)
        sanitized = re.sub(r'[A-Z]:\\[^\s]+', '[path]', sanitized)
        # Remove email addresses
        sanitized = re.sub(r'\S+@\S+', '[email]', sanitized)
        return f"{error_type}: {sanitized}"
    
    return client_message

