# CV Caddy - Resume Chatbot

A local, privacy-focused chatbot that answers questions about your resume using Retrieval Augmented Generation (RAG). Built with Ollama for local LLM inference and embeddings, this tool helps you prepare for interviews by practicing conversations about your professional background.

## Features

- **Local & Private**: All processing happens locally using Ollama - your resume data never leaves your machine
- **RAG-Powered**: Uses semantic search to find relevant resume sections for each question
- **PDF Support**: Automatically extracts text from PDF resumes
- **Personal Context**: Supports additional personal information files (Markdown or text)
- **Conversation History**: Maintains context across the conversation with intelligent summarization
- **Configurable Prompts**: Customize the chatbot's personality and behavior via `config.ini`
- **Caching**: Processes and caches embeddings for faster subsequent runs
- **Multiple Modes**: Interactive chat, SMS-style messaging, and telnet server modes
- **Security Hardened**: Input validation, rate limiting, prompt injection protection, and secure file handling
- **Daemon Support**: Run as a background service with reload and graceful shutdown
- **Systemd Integration**: Easy systemd service setup for production deployments

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages:
  - `ollama` - For LLM and embedding generation
  - `PyMuPDF` (fitz) - For PDF processing
  - `numpy` - For vector operations

## Installation

1. **Install Ollama** (if not already installed):
   ```bash
   # Visit https://ollama.ai/ for installation instructions
   # Or use: curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull required Ollama models**:
   ```bash
   ollama pull llama3.1:8b  # Or your preferred model
   ollama pull nomic-embed-text
   ```

3. **Install Python dependencies**:
   ```bash
   pip install ollama pymupdf numpy
   ```

   Or create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install ollama pymupdf numpy
   ```

4. **Start Ollama service** (if not running as a service):
   ```bash
   ollama serve
   ```

## Configuration

### Required Files

1. **Resume PDF**: Place your resume as `YOUR_RESUME.pdf` in the project directory
2. **Personal Info** (optional): Create `personal_info.md` or `personal_info.txt` with additional context about yourself

### Config File

The program will automatically create a `config.ini` file on first run. The configuration includes multiple sections:

#### [Files]
- `ResumePdfPath`: Path to your resume PDF file
- `PersonalInfoTxtPath`: Path to personal info text file (optional)
- `PersonalInfoMdPath`: Path to personal info markdown file (optional, preferred)
- `DataDir`: Directory for storing processed data (chunks and embeddings)

#### [Models]
- `LlmModel`: Ollama model for LLM generation (default: `llama3.1:8b`)
- `EmbeddingModel`: Ollama model for embeddings (default: `nomic-embed-text`)
- `Temperature`: Temperature for response generation (0.0-2.0, default: 0.8)
- `TopP`: Top-p (nucleus sampling) parameter (default: 0.9)

#### [Context]
- `MaxHistoryTokens`: Maximum tokens for conversation history (default: 2000)
- `MinRecentMessages`: Always keep at least this many recent messages (default: 6)
- `SummaryThreshold`: Summarize when history exceeds this fraction of max tokens (default: 0.8)

#### [RAG]
- `ChunkSize`: Size of text chunks for embedding in characters (default: 1500)
- `ChunkOverlap`: Overlap between chunks in characters (default: 200)
- `TopKChunks`: Number of top relevant chunks to retrieve (default: 3)

#### [Chatbot]
- `SystemPrompt`: System prompt that defines the chatbot's behavior and personality
- `InitialGreeting`: Template for initial greeting (supports `${NAME}` placeholder)
- `ShutdownMessage`: Message sent to active connections when daemon shuts down

#### [Security]
- `BindAddress`: Network binding address (default: `127.0.0.1` for localhost only)
- `MaxRequestsPerMinute`: Maximum requests per minute per IP address (default: 60)
- `MaxConnectionsPerIP`: Maximum concurrent connections per IP (default: 10)
- `ConnectionTimeout`: Connection timeout in seconds (default: 30.0)
- `IdleTimeout`: Idle timeout in seconds (default: 300.0)

Example `config.ini`:
```ini
[Chatbot]
SystemPrompt = 
	You are having a natural, professional conversation with a recruiter or hiring manager. 
	You ARE the person described in your background information below.
	
	Communication style:
	- Speak naturally and conversationally, as you would in a real interview
	- Use first person ('I', 'my', 'me') - you are talking about yourself
	- Be authentic and personable, not robotic or overly formal

InitialGreeting = Hi, my name is ${NAME}! Are you here to interview me?

[Security]
BindAddress = 127.0.0.1
MaxRequestsPerMinute = 60
MaxConnectionsPerIP = 10
```

## Usage

### Basic Interactive Mode

```bash
python main.py
```

The program will:
1. Check for existing processed data in the `data/` directory
2. If not found, process your resume and personal info files
3. Start an interactive chat session

### SMS/RCS Style Mode

For a more modern messaging interface with typing indicators:

```bash
python main.py --sms
# or
python main.py -s
```

### Telnet Server Mode

Run as a telnet server for remote access:

```bash
python main.py --telnet
# or
python main.py -t
```

Specify a custom port:
```bash
python main.py --telnet --port 8080
# or
python main.py -t -p 8080
```

### Daemon Mode

Run as a background daemon process (requires telnet mode):

```bash
python main.py --telnet --daemon
# or
python main.py -t -d
```

The daemon will:
- Fork into the background
- Write a PID file (default: `/var/run/cv-caddy.pid` if root, `./cv-caddy.pid` otherwise)
- Redirect output to log files (default: `cv-caddy-daemon.log` and `cv-caddy-daemon-error.log` in current directory)
- Continue running until stopped

**Custom PID file location:**
```bash
python main.py --telnet --daemon --pid-file /path/to/custom.pid
```

**Managing the daemon:**

Stop the daemon:
```bash
python main.py --stop
```

Reload configuration and reprocess resume:
```bash
python main.py --reload
```

### Systemd Service

Create a systemd service file for production deployment:

```bash
sudo python main.py --add-systemd
```

This will create `/etc/systemd/system/cv-caddy.service`. Then:

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable cv-caddy

# Start the service
sudo systemctl start cv-caddy

# Check status
sudo systemctl status cv-caddy

# View systemd logs
sudo journalctl -u cv-caddy -f

# Stop the service
sudo systemctl stop cv-caddy

# Restart the service
sudo systemctl restart cv-caddy

# Reload configuration (sends SIGHUP)
sudo systemctl reload cv-caddy
```

**Note**: When running under systemd, the `--daemon` flag is automatically detected and skipped (systemd handles process management).

### Reprocessing Resume

To force reprocessing of your resume (useful after updating files):

```bash
python main.py --reprocess
# or
python main.py -r
```

### Chat Commands

- Type your question and press Enter
- Type `quit`, `exit`, `bye`, or similar to exit
- Press `Ctrl+C` to exit (in interactive modes)

## Command-Line Options

| Flag | Short | Description |
|------|-------|-------------|
| `--reprocess` | `-r` | Force reprocessing of resume and context files |
| `--sms` | `-s` | Enable SMS/RCS messaging mode with typing indicators |
| `--telnet` | `-t` | Enable telnet server mode |
| `--port` | `-p` | Port number for telnet server (default: 2323) |
| `--daemon` | `-d` | Run as a daemon (background process). Requires `--telnet` |
| `--pid-file` | | Path to PID file (default: `/var/run/cv-caddy.pid` if root, `./cv-caddy.pid` otherwise) |
| `--reload` | | Send SIGHUP to running daemon to reload configuration and reprocess resume |
| `--stop` | | Send SIGTERM to running daemon to stop gracefully |
| `--add-systemd` | | Create systemd service file (requires root/sudo) |

## How It Works

1. **Processing Phase**:
   - Extracts text from your PDF resume
   - Reads personal info from `personal_info.md` or `personal_info.txt` (if present)
   - Validates all file paths for security
   - Splits the combined text into overlapping chunks
   - Generates embeddings for each chunk using the embedding model
   - Saves chunks and embeddings to the `data/` directory

2. **Query Phase**:
   - Validates and sanitizes user input
   - Checks for prompt injection attempts
   - Converts the question into an embedding
   - Finds the top K most relevant resume chunks using cosine similarity
   - Builds a prompt with system instructions, conversation history, relevant context, and the question
   - Generates a response using the LLM model
   - Updates conversation history for context (with automatic summarization when needed)

3. **Security**:
   - Input length limits (max 5000 characters)
   - Input sanitization (removes control characters)
   - Prompt injection detection and mitigation
   - Rate limiting (per IP address)
   - Connection limits (per IP address)
   - File path validation (prevents path traversal)
   - Error message sanitization (prevents information disclosure)

## File Structure

```
cv-caddy/
├── main.py                    # Main program
├── config.py                  # Configuration defaults
├── config.ini                 # System configuration (auto-generated)
├── daemon_utils.py            # Daemon utilities
├── security.py                # Security utilities
├── resume_processor.py        # Resume processing and embedding
├── rag.py                     # RAG (retrieval) functions
├── context_manager.py         # Conversation history management
├── recruiter_tracker.py       # Recruiter information extraction
├── text_processing.py         # Text processing utilities
├── YOUR_RESUME.pdf            # Your resume (required)
├── personal_info.md           # Additional personal info (optional, preferred)
├── personal_info.txt          # Additional personal info (optional, fallback)
├── data/                      # Processed data (auto-generated)
│   ├── chunks.json            # Text chunks
│   ├── embeddings.npy         # Embedding vectors
│   └── chunk_metadata.json    # Metadata (name, personal info indices)
└── cv-caddy.pid               # PID file (when running as daemon, if not root)
```

## Security Features

The application includes comprehensive security measures:

- **Input Validation**: Maximum input length limits and sanitization
- **Prompt Injection Protection**: Detection and mitigation of prompt injection attempts
- **Rate Limiting**: Configurable requests per minute per IP address
- **Connection Limits**: Maximum concurrent connections per IP address
- **Path Traversal Protection**: All file paths are validated to prevent directory traversal attacks
- **Error Sanitization**: Error messages are sanitized to prevent information disclosure
- **Secure Bind Address**: Defaults to localhost-only (127.0.0.1) to prevent network exposure

## Troubleshooting

**Error: "Is Ollama running?"**
- Make sure Ollama is installed and the service is running: `ollama serve`
- Check if Ollama is accessible: `ollama list`

**Error: "Model not found"**
- Pull the required models: `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`
- Update `config.ini` if using different models

**Resume not found**
- Ensure your resume is named correctly or update `ResumePdfPath` in `config.ini`
- Check that the file path is valid and accessible

**Outdated responses after updating resume**
- Use the `--reprocess` flag to regenerate embeddings: `python main.py --reprocess`

**Connection refused (telnet mode)**
- Check that the service is running: `sudo systemctl status cv-caddy`
- Verify the bind address in `config.ini` (should be `127.0.0.1` for localhost)
- Check firewall settings if binding to `0.0.0.0`
- View systemd logs: `sudo journalctl -u cv-caddy -f` (for systemd) or check daemon log files

**Daemon won't start**
- Check if another instance is already running: `python main.py --stop`
- Verify PID file permissions
- Check daemon error log file (default: `cv-caddy-daemon-error.log`)

**Systemd service keeps restarting**
- Check journal logs: `sudo journalctl -u cv-caddy -n 100`
- Ensure the service file doesn't have `--daemon` flag (systemd handles process management)
- Verify file paths and permissions in the service file
- Check that the user specified in the service file has proper permissions

**Rate limit errors**
- Adjust `MaxRequestsPerMinute` in `config.ini` if legitimate traffic is being blocked
- Check if multiple clients are connecting from the same IP

## Advanced Usage

### Custom Initial Greeting

Edit `InitialGreeting` in `config.ini`:
```ini
[Chatbot]
InitialGreeting = Hello! I'm ${NAME}. Ready to discuss my background!
```

The `${NAME}` placeholder will be replaced with the name extracted from your resume.

### Adjusting Model Parameters

Edit the `[Models]` section in `config.ini`:
```ini
[Models]
LlmModel = llama3.1:8b
Temperature = 0.8
TopP = 0.9
```

### Network Configuration

For remote access (use with caution):
```ini
[Security]
BindAddress = 0.0.0.0  # Listen on all interfaces
MaxRequestsPerMinute = 120
MaxConnectionsPerIP = 20
```

**Warning**: Binding to `0.0.0.0` exposes the server to all network interfaces. Only do this on trusted networks and consider additional security measures.

### Reload Without Restart

When running as a daemon or systemd service, you can reload configuration without restarting:
```bash
# For daemon mode
python main.py --reload

# For systemd
sudo systemctl reload cv-caddy
```

This will:
- Reload `config.ini`
- Reprocess the resume
- Restart the server with new configuration
- Active connections will continue until they naturally end

## License

This project is provided as-is for personal use.
