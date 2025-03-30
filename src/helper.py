import os
import pickle
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Updated import for Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
import langdetect
from langchain.callbacks.base import BaseCallbackHandler
import os
import pickle
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # Updated import for Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
import langdetect

# Configure warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure model parameters
SUPPORTED_LANGUAGES = ["en", "sa", "hi"]
DEFAULT_EMB_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TEMPERATURE = 0.2
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
class StreamHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to Streamlit."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")


def detect_language(text: str) -> str:
    """
    Detect language of the text and return language code.
    Specializes in detecting Sanskrit against other Indian languages.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Language code: 'sa' for Sanskrit, 'hi' for Hindi, 'en' for English
    """
    try:
        lang = langdetect.detect(text[:1000])
        # For Sanskrit, we need additional rules as langdetect might confuse it
        if lang in ['hi', 'mr', 'ne', 'sa']:
            # Sanskrit-specific character patterns
            sanskrit_patterns = ['ः', 'ॐ', 'ऋ', 'ॠ', 'ॡ', 'ङ्', 'ञ्', 'क्ष्']
            sanskrit_count = sum(text.count(pattern) for pattern in sanskrit_patterns)
            
            if sanskrit_count > 5:
                return 'sa'
            
        return lang if lang in SUPPORTED_LANGUAGES else 'en'
    except:
        return 'en'


def get_document_loaders(file_paths: List[str]) -> Dict[str, Any]:
    """
    Create document loaders for different file types.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        Dictionary with file paths as keys and document loaders as values
    """
    loaders = {}
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loaders[file_path] = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loaders[file_path] = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            loaders[file_path] = TextLoader(file_path)
        elif file_path.endswith(('.md', '.markdown')):
            loaders[file_path] = UnstructuredMarkdownLoader(file_path)
    return loaders


def process_documents(
    file_paths: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = False
) -> Tuple[List[Document], Dict[str, str]]:
    """
    Process documents and split them into chunks.
    
    Args:
        file_paths: List of file paths
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        verbose: Whether to print debug information
        
    Returns:
        Tuple of processed documents and metadata about document languages
    """
    loaders = get_document_loaders(file_paths)
    
    if verbose:
        st.write(f"Processing {len(loaders)} documents")
    
    all_docs = []
    doc_languages = {}
    
    # Process each document
    for file_path, loader in loaders.items():
        try:
            # Load document
            file_docs = loader.load()
            
            # Detect language for each document
            full_text = " ".join([doc.page_content for doc in file_docs])
            doc_lang = detect_language(full_text)
            doc_languages[file_path] = doc_lang
            
            if verbose:
                st.write(f"Detected language for {os.path.basename(file_path)}: {doc_lang}")
            
            # Extract file metadata
            file_name = os.path.basename(file_path)
            file_type = os.path.splitext(file_path)[1]
            
            # Enrich metadata
            for doc in file_docs:
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                
                doc.metadata.update({
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": file_type,
                    "language": doc_lang,
                })
                
                # Add page number and section metadata if available
                if 'page' in doc.metadata:
                    doc.metadata['section'] = f"Page {doc.metadata['page']}"
            
            all_docs.extend(file_docs)
        except Exception as e:
            if verbose:
                st.error(f"Error processing {file_path}: {str(e)}")
    
    # Split documents into chunks
    if verbose:
        st.write(f"Splitting {len(all_docs)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    
    split_docs = []
    for doc in all_docs:
        chunks = text_splitter.split_documents([doc])
        # Preserve parent metadata in chunks
        for chunk in chunks:
            if hasattr(doc, 'metadata'):
                chunk.metadata.update(doc.metadata)
        split_docs.extend(chunks)
    
    if verbose:
        st.write(f"Created {len(split_docs)} chunks")
    
    return split_docs, doc_languages


def get_embeddings(model_name: str = DEFAULT_EMB_MODEL) -> HuggingFaceEmbeddings:
    """
    Get embedding model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        HuggingFaceEmbeddings model
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstore(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Optional[str] = None
) -> FAISS:
    """
    Get vector store for documents.
    
    Args:
        documents: List of documents
        embeddings: Embedding model
        persist_directory: Directory to persist vector store
        
    Returns:
        FAISS vector store
    """
    if persist_directory and os.path.exists(os.path.join(persist_directory, "index.faiss")):
        vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            vectorstore.save_local(persist_directory)
    
    return vectorstore

def get_conversation_chain(
    vectorstore: FAISS,
    model_name: str = "llama3-70b-8192",
    temperature: float = DEFAULT_TEMPERATURE,
    stream_handler: Optional[StreamHandler] = None
) -> ConversationalRetrievalChain:
    """
    Get conversation chain for chat with documents.
    
    Args:
        vectorstore: Vector store
        model_name: Name of the LLM model
        temperature: Temperature for LLM
        stream_handler: Callback handler for streaming
        
    Returns:
        ConversationalRetrievalChain
    """
    # Create LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        streaming=stream_handler is not None,
        callbacks=[stream_handler] if stream_handler else None,
    )
    
    # Create memory with specified output_key
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Specify the output key to store in memory
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # System template with context variable
    system_template = """
    You are a knowledgeable assistant specializing in ancient Hindu and Sanskrit religious texts including the Mahabharata, Ramayana, Bhagavad Gita, Upanishads, Vedas, and Puranas.

    When answering queries:
    1. Provide clear, accurate, and engaging responses
    2. Use natural, human-like language
    3. Adapt your tone to be friendly and approachable
    4. Offer thorough explanations without overwhelming the user
    5. Include relatable examples when appropriate
    6. Cite specific verses or chapters when available
    7. Present multiple interpretative perspectives when relevant
    8. Maintain cultural sensitivity and respect for these sacred texts
    
    Format your responses like this example:
    
    **Question: Did Sita enter the fire?**
    
    Yes, in the Ramayana, Sita underwent *Agni Pariksha* (fire ordeal) after being rescued from Lanka to prove her purity. Lord Agni (fire god) protected her, demonstrating her divine chastity.
    
    The Agni Pariksha occurs in the Yuddha Kanda of the Ramayana, where Sita voluntarily enters the fire to prove her purity after Rama expresses doubt following her rescue from Ravana's captivity. When she emerges unharmed, it confirms her chastity and devotion.
    
    This episode represents themes of:
    - Devotion and sacrifice
    - Social expectations and gender roles
    - Divine protection for the righteous
    - The complex nature of dharma (duty)
    
    Use the provided context to answer questions accurately. If information is not available in the context, state this clearly rather than making up information.
    
    Context:
    {context}
    """
    
    # Import necessary components for the LangChain prompt
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, StuffDocumentsChain
    
    # Define the prompt template with the required variables
    prompt = PromptTemplate(
        template=system_template + "\n\nQuestion: {question}\n",
        input_variables=["context", "question"]
    )
    
    # Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Create StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    
    # Create question generator prompt
    question_prompt = PromptTemplate(
        template="Given the conversation history and the latest user question, rephrase the question to be a standalone question that captures all relevant context from the conversation history.\n\nChat History:\n{chat_history}\n\nLatest Question: {question}\n\nStandalone question:",
        input_variables=["chat_history", "question"]
    )
    
    # Create question generator chain
    question_generator = LLMChain(llm=llm, prompt=question_prompt)
    
    # Create the conversational chain
    conversation_chain = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=stuff_chain,
        question_generator=question_generator,
        memory=memory,
        return_source_documents=True,
    )
    
    return conversation_chain

def format_docs(docs: List[Document]) -> str:
    """
    Format document snippets for display.
    
    Args:
        docs: List of documents
        
    Returns:
        Formatted string
    """
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        file_name = doc.metadata.get("file_name", os.path.basename(source))
        language = doc.metadata.get("language", "Unknown")
        section = doc.metadata.get("section", "")
        
        formatted_doc = f"**Source {i+1}:** {file_name} ({language})\n"
        if section:
            formatted_doc += f"**Section:** {section}\n"
        formatted_doc += f"**Content:** {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}\n\n"
        
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def save_uploaded_file(uploaded_file, directory="./uploads"):
    """
    Save uploaded file to directory.
    
    Args:
        uploaded_file: Uploaded file
        directory: Directory to save file
        
    Returns:
        Path to saved file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def clear_session_states():
    """Clear all session state variables related to conversation."""
    if 'conversation' in st.session_state:
        del st.session_state['conversation']
    if 'chat_history' in st.session_state:
        del st.session_state['chat_history']
    if 'documents' in st.session_state:
        del st.session_state['documents']
    if 'document_languages' in st.session_state:
        del st.session_state['document_languages']
    if 'vectorstore' in st.session_state:
        del st.session_state['vectorstore']