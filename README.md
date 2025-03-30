# JnanaSetu

## Overview

The JnanaSetu is a Retrieval-Augmented Generation (RAG) system designed to provide intelligent, contextual responses to questions about ancient Hindu and Sanskrit religious texts. This application leverages LLM technology via Groq, along with vector embeddings and retrieval mechanisms to enable accurate and insightful conversations about texts such as the Mahabharata, Ramayana, Bhagavad Gita, Upanishads, Vedas, and Puranas.

## Features

- **Multi-document Processing**: Upload and process PDF, DOCX, TXT, and Markdown files containing religious texts
- **Multi-language Support**: Handles English, Sanskrit, and Hindi texts with specialized language detection
- **Intelligent Chunking**: Splits documents into optimal chunks while preserving context
- **Vector Embeddings**: Creates semantic embeddings using HuggingFace models
- **Contextual Conversations**: Maintains conversation history to provide coherent and contextual responses
- **Citation Support**: References source documents and sections for each response
- **Streaming Responses**: Shows real-time generation of AI responses
- **Customizable Parameters**: Configure chunk size, model selection, and temperature settings

## Requirements

- Python 3.8+
- Groq API key
- Dependencies as listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sanskrit-texts-assistant.git
   cd sanskrit-texts-assistant
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the URL provided in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Upload documents (PDF, DOCX, TXT, MD)
   - Configure advanced settings (optional)
   - Process the documents

4. Once documents are processed, you can:
   - Ask questions about the religious texts in the chat interface
   - View source documents used to generate responses
   - See document information and language distribution

## Directory Structure

```
sanskrit-texts-assistant/
├── app.py                # Main Streamlit application
├── src/
│   ├── __init__.py
│   └── helper.py         # Helper functions for document processing and RAG
├── uploads/              # Directory for uploaded documents (created automatically)
├── .env                  # Environment variables
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Advanced Settings

### Document Processing

- **Chunk Size**: Controls the size of text chunks (default: 1000)
- **Chunk Overlap**: Controls the overlap between chunks for context preservation (default: 100)
- **Embedding Model**: Choose from several multilingual sentence transformer models

### LLM Configuration

- **LLM Model**: Select from available Groq models (llama3-70b-8192, llama3-8b-8192, etc.)
- **Temperature**: Adjust the creativity vs. determinism of responses (default: 0.2)
- **Verbose Mode**: Enable detailed logging for debugging purposes

## How It Works

1. **Document Upload**: Users upload religious texts in various formats
2. **Text Extraction**: The system extracts text content from different file formats
3. **Language Detection**: Each document's language is detected (English, Sanskrit, or Hindi)
4. **Chunking**: Documents are split into manageable chunks with appropriate overlap
5. **Embedding**: Text chunks are converted to vector embeddings
6. **Vector Storage**: Embeddings are stored in a FAISS vector database
7. **Query Processing**: User questions are analyzed and relevant document chunks are retrieved
8. **Response Generation**: The LLM generates responses based on retrieved context
9. **Citation**: Source documents are tracked and can be displayed to the user

## Customizing the Assistant

The system prompt in `helper.py` can be modified to adjust the assistant's personality, expertise level, or response format. The current prompt is optimized for providing clear, accurate, and engaging responses about Hindu and Sanskrit religious texts.

## Troubleshooting

- **API Key Issues**: Ensure your Groq API key is correctly set in the `.env` file
- **Memory Errors**: For large documents, consider increasing your system's available memory
- **Performance Issues**: Adjust chunk size and overlap parameters for optimal performance
- **Language Detection**: For specialized Sanskrit texts, you may need to adjust the language detection logic

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the framework
- Groq for the LLM API
- HuggingFace for embedding models
- Streamlit for the user interface
