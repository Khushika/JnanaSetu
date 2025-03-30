import os
import streamlit as st
from dotenv import load_dotenv
import time
import pandas as pd
from src.helper import (
    get_conversation_chain,
    get_embeddings,
    get_vectorstore,
    process_documents,
    save_uploaded_file,
    clear_session_states,
    StreamHandler,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TEMPERATURE,
    DEFAULT_EMB_MODEL,
)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="JnanaSetu - Bridging Seekers with Knowledge ",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.bot {
        background-color: #e6f3ff;
    }
    .chat-message .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex: 1;
    }
    .stButton button {
        width: 100%;
    }
    .source-documents {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def display_chat_message(role, content, avatar):
    """Display a chat message with avatar."""
    with st.container():
        col1, col2 = st.columns([1, 12])
        with col1:
            st.image(avatar, width=50)
        with col2:
            st.markdown(content)
        st.markdown("---")


def handle_userinput(query):
    """Process user query and get response from the conversation chain."""
    if 'conversation' not in st.session_state:
        st.error("Please upload and process documents first.")
        return
    
    # Store user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Create streaming response container
    response_container = st.empty()
    stream_handler = StreamHandler(response_container)
    
    # Update conversation chain with stream handler
    st.session_state.conversation.callbacks = [stream_handler]
    
    try:
        # Get response
        with st.spinner("Thinking..."):
            # Add timeout to prevent hanging
            response = st.session_state.conversation({
                "question": query,
                "chat_history": st.session_state.chat_history
            })
            
            # Debug info (optional)
            st.write(f"Response keys: {response.keys()}")
        
        # Store response content
        answer = response.get("answer", "No answer received from the model")
        source_documents = response.get("source_documents", [])
        
        # Store bot message with full answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Update chat history
        st.session_state.chat_history.append((query, answer))
        
        # Store source documents for this query
        if "source_history" not in st.session_state:
            st.session_state.source_history = {}
        st.session_state.source_history[query] = source_documents
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        # Log the full error for debugging
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def display_chat_history():
    """Display chat history."""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            display_chat_message(
                "user", 
                f"**You:** {content}", 
                "https://api.dicebear.com/7.x/bottts/svg?seed=user"
            )
        else:
            display_chat_message(
                "assistant", 
                f"**Assistant:** {content}", 
                "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"
            )


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "document_languages" not in st.session_state:
        st.session_state.document_languages = {}


def main():
    """Main function."""
    init_session_state()
    
    # Define sidebar
    with st.sidebar:
        st.title("🕉️ Sanskrit Texts RAG")
        st.markdown("---")
        
        # File upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload religious texts (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )
        
        # Advanced options
        with st.expander("Advanced Settings", expanded=False):
            # Chunk settings
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=4000,
                    value=DEFAULT_CHUNK_SIZE,
                    step=100,
                )
            with col2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=1000,
                    value=DEFAULT_CHUNK_OVERLAP,
                    step=10,
                )
            
            # Embedding model
            emb_model = st.selectbox(
                "Embedding Model",
                [
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ],
                index=0,
            )
            
            # LLM settings
            llm_model = st.selectbox(
                "LLM Model",
                [
                    "llama3-70b-8192",
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma-7b-it",
                ],
                index=0,
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_TEMPERATURE,
                step=0.1,
            )
            
            # Verbose mode
            verbose = st.checkbox("Verbose Mode", value=False)
            
            st.caption("Higher chunk size processes fewer chunks but preserves more context. Higher temperature makes responses more creative but potentially less accurate.")
        
        # Process files button
        process_btn = st.button("Process Documents", type="primary")
        
        if uploaded_files and process_btn:
            with st.spinner("Processing documents..."):
                # Clear previous conversation
                clear_session_states()
                st.session_state.messages = []
                st.session_state.chat_history = []
                
                # Save uploaded files
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = save_uploaded_file(uploaded_file)
                    file_paths.append(file_path)
                
                # Process documents
                start_time = time.time()
                
                progress_bar = st.progress(0)
                st.markdown("Processing documents...")
                
                # Step 1: Process documents
                progress_bar.progress(10)
                documents, doc_languages = process_documents(
                    file_paths=file_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    verbose=verbose,
                )
                
                # Step 2: Get embeddings
                progress_bar.progress(30)
                st.markdown("Creating embeddings...")
                embeddings = get_embeddings(model_name=emb_model)
                
                # Step 3: Create vector store
                progress_bar.progress(60)
                st.markdown("Building vector database...")
                vectorstore = get_vectorstore(documents, embeddings)
                
                # Step 4: Create conversation chain
                progress_bar.progress(80)
                st.markdown("Initializing conversation model...")
                conversation = get_conversation_chain(
                    vectorstore=vectorstore,
                    model_name=llm_model,
                    temperature=temperature,
                )
                
                # Step 5: Save to session state
                progress_bar.progress(100)
                
                # Store in session state
                st.session_state.documents = documents
                st.session_state.document_languages = doc_languages
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = conversation
                
                processing_time = time.time() - start_time
                
                # Summary
                st.success(f"✅ Processed {len(file_paths)} documents into {len(documents)} chunks in {processing_time:.2f} seconds")
                
                # Show document languages
                if verbose:
                    st.subheader("Document Languages")
                    lang_df = pd.DataFrame(
                        [(os.path.basename(k), v) for k, v in doc_languages.items()], 
                        columns=["Document", "Language"]
                    )
                    st.dataframe(lang_df)
        
        # Display document info
        if "documents" in st.session_state and st.session_state.documents:
            st.markdown("---")
            st.subheader("Document Information")
            st.markdown(f"📚 **Documents:** {len(set([doc.metadata.get('source', '') for doc in st.session_state.documents]))}")
            st.markdown(f"📄 **Chunks:** {len(st.session_state.documents)}")
            
            # Language distribution
            if "document_languages" in st.session_state:
                lang_counts = {}
                for lang in st.session_state.document_languages.values():
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                
                for lang, count in lang_counts.items():
                    lang_display = {
                        "en": "English",
                        "hi": "Hindi",
                        "sa": "Sanskrit",
                        "te": "Telugu",
                        "ta": "Tamil",
                        "ka": "Kannada",
                        "ma": "Malayalam"
                    }.get(lang, lang)
                    st.markdown(f"🔤 **{lang_display}:** {count} documents")
        
        # Footer
        st.markdown("---")
        st.caption("Built with LangChain, Groq, and Streamlit")
    
    # Main chat area
    st.header("Sanskrit Texts Knowledge Assistant")
    
    # Check if documents are processed
    if "documents" not in st.session_state or not st.session_state.documents:
        st.info("👈 Please upload and process documents using the sidebar to start chatting.")
        
        # Sample questions
        st.markdown("### Sample Questions You Can Ask:")
        sample_questions = [
            "What is the significance of Arjuna's dilemma in the Bhagavad Gita?",
            "Explain the concept of Dharma in the Mahabharata",
            "What does the Ramayana teach about loyalty and righteousness?",
            "Compare the philosophy of Advaita Vedanta with Dvaita in the Upanishads",
            "What is the symbolism behind Hanuman's role in the Ramayana?"
        ]
        
        for q in sample_questions:
            st.markdown(f"- {q}")
        
        return
    
    # Chat interface
    query = st.chat_input("Ask a question about the texts...")
    if query:
        handle_userinput(query)
    
    # Display chat history
    display_chat_history()
    
    # Show source documents for the last query
    if "source_history" in st.session_state and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else None
        
        if last_query and last_query in st.session_state.source_history:
            with st.expander("View Source Documents", expanded=False):
                source_docs = st.session_state.source_history[last_query]
                
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get("source", "Unknown")
                    file_name = doc.metadata.get("file_name", os.path.basename(source))
                    language = doc.metadata.get("language", "Unknown")
                    section = doc.metadata.get("section", "")
                    
                    st.markdown(f"**Source {i+1}:** {file_name} ({language})")
                    if section:
                        st.markdown(f"**Section:** {section}")
                    
                    st.markdown("**Content:**")
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                    st.markdown("---")


if __name__ == "__main__":
    main()
