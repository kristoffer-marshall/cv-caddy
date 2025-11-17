"""
Configuration constants and default values for the resume chatbot.
"""

# Configuration file name
CONFIG_FILE = "config.ini"

# Default values (used as fallbacks if config.ini is missing or incomplete)
DEFAULT_RESUME_PDF_PATH = "YOUR_RESUME.pdf"
DEFAULT_PERSONAL_INFO_TXT_PATH = "personal_info.txt"
DEFAULT_PERSONAL_INFO_MD_PATH = "personal_info.md"
DEFAULT_DATA_DIR = "data"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
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

