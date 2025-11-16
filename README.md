# Resume Chatbot

A local, privacy-focused chatbot that answers questions about your resume using Retrieval Augmented Generation (RAG). Built with Ollama for local LLM inference and embeddings, this tool helps you prepare for interviews by practicing conversations about your professional background.

## Features

- **Local & Private**: All processing happens locally using Ollama - your resume data never leaves your machine
- **RAG-Powered**: Uses semantic search to find relevant resume sections for each question
- **PDF Support**: Automatically extracts text from PDF resumes
- **Personal Context**: Supports additional personal information files (Markdown or text)
- **Conversation History**: Maintains context across the conversation
- **Configurable Prompts**: Customize the chatbot's personality and behavior via `config.ini`
- **Caching**: Processes and caches embeddings for faster subsequent runs

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
   ollama pull llama3
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

The program will automatically create a `config.ini` file on first run. You can customize the system prompt to change how the chatbot behaves:

```ini
[Chatbot]
SystemPrompt = 
    You are a professional vulnerability management engineer who is talking to a recruiter or hiring manager.
    Do not make up information. If the answer is not in the context, politely state that you are unsure, but can get back to them later.
    Use the knowledge from your resume, but do no reference your resume in conversation. Act casual.
```

Edit `config.ini` to match your profession and desired conversation style.

## Usage

### Basic Usage

```bash
python resume-chatbot.py
```

The program will:
1. Check for existing processed data in the `data/` directory
2. If not found, process your resume and personal info files
3. Start an interactive chat session

### Reprocessing

To force reprocessing of your resume (useful after updating files):

```bash
python resume-chatbot.py --reprocess
# or
python resume-chatbot.py -r
```

### Chat Commands

- Type your question and press Enter
- Type `quit` to exit
- Press `Ctrl+C` to exit

## How It Works

1. **Processing Phase**:
   - Extracts text from your PDF resume
   - Reads personal info from `personal_info.md` or `personal_info.txt` (if present)
   - Splits the combined text into overlapping chunks
   - Generates embeddings for each chunk using `nomic-embed-text`
   - Saves chunks and embeddings to the `data/` directory

2. **Query Phase**:
   - Converts your question into an embedding
   - Finds the top 3 most relevant resume chunks using cosine similarity
   - Builds a prompt with system instructions, conversation history, relevant context, and your question
   - Generates a response using `llama3` model
   - Updates conversation history for context

## File Structure

```
resume-chatbot/
├── resume-chatbot.py      # Main program
├── config.ini             # System prompt configuration (auto-generated)
├── YOUR_RESUME.pdf        # Your resume (required)
├── personal_info.md       # Additional personal info (optional, preferred)
├── personal_info.txt      # Additional personal info (optional, fallback)
└── data/                  # Processed data (auto-generated)
    ├── chunks.json        # Text chunks
    └── embeddings.npy     # Embedding vectors
```

## Customization

### Changing Models

Edit the constants at the top of `resume-chatbot.py`:

```python
LLM_MODEL = "llama3"              # Change to any Ollama model
EMBEDDING_MODEL = "nomic-embed-text"  # Change to any embedding model
```

### Adjusting Chunk Size

Modify the chunk parameters in the `split_text_into_chunks` function call:

```python
text_chunks = split_text_into_chunks(full_text, chunk_size=1500, overlap=200)
```

### Changing Number of Relevant Chunks

Modify the `k` parameter in the `find_relevant_chunks` call:

```python
relevant_chunks = find_relevant_chunks(
    query_embedding, 
    all_embeddings, 
    text_chunks,
    k=3  # Change this number
)
```

## Troubleshooting

**Error: "Is Ollama running?"**
- Make sure Ollama is installed and the service is running: `ollama serve`

**Error: "Model not found"**
- Pull the required models: `ollama pull llama3` and `ollama pull nomic-embed-text`

**Resume not found**
- Ensure your resume is named exactly `YOUR_RESUME.pdf` in the project directory
- Or update the `RESUME_PDF_PATH` constant in the script

**Outdated responses after updating resume**
- Use the `--reprocess` flag to regenerate embeddings

## License

This project is provided as-is for personal use.

