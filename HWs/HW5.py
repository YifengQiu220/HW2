# HW5.py — Enhanced Chatbot with Function-based Retrieval

import os
import glob
import textwrap
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI

# Optional providers (guarded imports)
try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---- SQLite fix (for Streamlit Cloud)
try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

# ===================== Constants =====================
PAGE_TITLE        = "HW5 – Enhanced iSchool Chatbot"
ORGS_FOLDER       = os.path.join("HWs", "hw4_orgs")    
CHROMA_PATH       = "./ChromaDB_hw4"                   
CHROMA_COLLECTION = "HW4Collection"                    
EMBED_MODEL       = "text-embedding-3-small"           
TOP_K             = 3
BUFFER_PAIRS      = 5  # Short-term memory

# ===================== UI Header =====================
st.title(PAGE_TITLE)
st.caption(
    "Enhanced chatbot using function-based retrieval and short-term memory. "
    "The retrieval function operates independently from the LLM prompting."
)

# ===================== Chunking Function =====================
def chunk_into_two(text: str) -> List[str]:
    """Split each document into exactly two chunks."""
    text = " ".join(text.split())
    if not text:
        return ["", ""]
    if len(text) <= 400:
        mid = len(text) // 2
        return [text[:mid].strip(), text[mid:].strip()]

    mid = len(text) // 2
    left = max(0, mid - 200)
    right = min(len(text), mid + 200)
    window = text[left:right]
    snap = window.rfind(". ")
    if snap != -1:
        cut = left + snap + 1
    else:
        cut = mid

    return [text[:cut].strip(), text[cut:].strip()]

# ===================== Build ChromaDB =====================
@st.cache_resource(show_spinner=True)
def build_chromadb() -> chromadb.Collection:
    """Create or load the persistent Chroma collection."""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to .streamlit/secrets.toml or environment.")
        st.stop()

    if not os.path.isdir(ORGS_FOLDER):
        st.error(f"Source folder not found: {os.path.abspath(ORGS_FOLDER)}")
        st.info("Create the folder and copy all HTML files from su_orgs.zip, then rerun.")
        st.stop()

    html_paths = sorted(glob.glob(os.path.join(ORGS_FOLDER, "*.html")))
    if not html_paths:
        st.error(f"No HTML files found in {os.path.abspath(ORGS_FOLDER)}")
        st.stop()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    coll = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )

    if coll.count() == 0:
        ids, docs, metas = [], [], []
        for p in html_paths:
            fname = os.path.basename(p)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
            except Exception as e:
                st.warning(f"Error reading {fname}: {e}")
                continue

            chunks = chunk_into_two(text)
            for idx, chunk in enumerate(chunks):
                ids.append(f"{fname}-{idx:04d}")
                docs.append(chunk)
                metas.append({"filename": fname, "chunk": idx})

        if ids:
            batch = 100
            for i in range(0, len(ids), batch):
                coll.add(
                    documents=docs[i:i+batch],
                    metadatas=metas[i:i+batch],
                    ids=ids[i:i+batch],
                )
            st.sidebar.success(f"Indexed {len(ids)} chunks from {len(html_paths)} HTML files")

    return coll

# Build or load the collection
if "HW5_vectorDB" not in st.session_state:
    st.session_state.HW5_vectorDB = build_chromadb()

# ===================== CORE FUNCTION FOR HW5 =====================
def get_relevant_club_info(query: str, k: int = TOP_K) -> str:
    """
    Main retrieval function as specified in HW5.
    Takes a query and returns relevant club/course information from vector search.
    
    Args:
        query: User's input question
        k: Number of top results to retrieve
    
    Returns:
        str: Retrieved relevant information from the vector database
    """
    try:
        collection = st.session_state.HW5_vectorDB
        results = collection.query(query_texts=[query], n_results=k)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not documents:
            return "No relevant information found in the database."
        
        # Format the retrieved information
        retrieved_info = []
        for doc, meta in zip(documents, metadatas):
            if doc:
                filename = meta.get("filename", "unknown")
                chunk_num = meta.get("chunk", 0)
                # Clean up the text
                clean_doc = " ".join(doc.split())[:500]  # Limit length for readability
                retrieved_info.append(
                    f"Source: {filename} (chunk {chunk_num})\n"
                    f"Content: {clean_doc}..."
                )
        
        return "\n\n".join(retrieved_info)
    
    except Exception as e:
        return f"Error during retrieval: {str(e)}"

# ===================== LLM Client Setup =====================
# We'll handle client initialization directly in the chat section like HW4

# ===================== Short-term Memory Management =====================
def get_buffered_messages(messages: List[Dict[str, str]], pairs: int = BUFFER_PAIRS) -> List[Dict[str, str]]:
    """
    Maintain short-term memory by keeping only recent conversation pairs.
    Keeps the initial greeting and the last N user-assistant exchanges.
    """
    if len(messages) <= 2:
        return messages
    
    # Keep first message (greeting) and last N pairs
    greeting = [messages[0]] if messages else []
    recent_pairs = messages[-(pairs * 2):]
    
    return greeting + recent_pairs

# ===================== Sidebar Configuration =====================
st.sidebar.header("Configuration")

# Model selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    ["openai", "anthropic", "gemini"],
    format_func=lambda x: x.title()
)

if provider == "openai":
    model_name = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
elif provider == "anthropic":
    model_name = st.sidebar.selectbox(
        "Model", 
        ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
        index=0
    )
else:  # gemini
    model_name = st.sidebar.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        index=0
    )

# Retrieval settings
st.sidebar.subheader("Retrieval Settings")
show_retrieval = st.sidebar.checkbox("Show retrieved context", value=True)
retrieval_method = st.sidebar.radio(
    "How to use retrieved info:",
    ["In system prompt", "As function result"],
    help="Choose how to pass retrieved information to the LLM"
)

# Memory settings
st.sidebar.subheader("Memory Settings")
st.sidebar.info(f"Keeping last {BUFFER_PAIRS} conversation pairs")

# Clear chat button
if st.sidebar.button("Clear Chat 🗑️"):
    st.session_state.messages = [
        {"role": "assistant", 
         "content": "Hi! I'm your enhanced iSchool assistant. Ask me about Syracuse iSchool student organizations!"}
    ]
    st.rerun()

# ===================== Initialize Chat State =====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! I'm your enhanced iSchool assistant. Ask me about Syracuse iSchool student organizations!"}
    ]

# ===================== Display Chat History =====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===================== Handle User Input =====================
if user_input := st.chat_input("Ask about student organizations..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # STEP 1: Call the retrieval function
    with st.spinner("Searching relevant information..."):
        relevant_info = get_relevant_club_info(user_input)
    
    # Optional: Display retrieved context
    if show_retrieval:
        with st.expander("📚 Retrieved Context", expanded=False):
            st.text(relevant_info)
    
    # STEP 2: Prepare messages for LLM based on selected method
    with st.chat_message("assistant"):
        # Show model info
        st.caption(f"🤖 {provider.title()} • {model_name} • Method: {retrieval_method}")
        
        # Build message list with short-term memory
        buffered_messages = get_buffered_messages(st.session_state.messages, BUFFER_PAIRS)
        
        # Prepare system message based on retrieval method
        if retrieval_method == "In system prompt":
            # Method 1: Include retrieval results in system prompt
            system_content = (
                "You are a helpful assistant for Syracuse iSchool. "
                "Answer questions based on the following retrieved information:\n\n"
                f"{relevant_info}\n\n"
                "If the information doesn't answer the question, say you don't know. "
                "Always cite the source files when using information from them."
            )
        else:
            # Method 2: Treat retrieval as a function result
            system_content = (
                "You are a helpful assistant for Syracuse iSchool. "
                "You have access to a search function that retrieved relevant information. "
                "Use this information to answer questions accurately."
            )
        
        # Construct messages for LLM
        messages_for_llm = [{"role": "system", "content": system_content}]
        
        if retrieval_method == "As function result":
            # Add the retrieval result as if it came from a function call
            messages_for_llm.append({
                "role": "assistant",
                "content": f"I searched for relevant information and found:\n\n{relevant_info}"
            })
            messages_for_llm.append({
                "role": "user",
                "content": f"Based on this information, please answer: {user_input}"
            })
        else:
            # Add buffered conversation history
            for msg in buffered_messages[1:]:  # Skip the greeting
                if msg["role"] in ["user", "assistant"]:
                    messages_for_llm.append(msg)
        
        # STEP 3: Call LLM and stream response
        try:
            if provider == "openai":
                api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("Missing OPENAI_API_KEY")
                    st.stop()
                client = OpenAI(api_key=api_key)
                
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages_for_llm,
                    temperature=0.7,
                    stream=True
                )
                response = st.write_stream(stream)
                
            elif provider == "anthropic":
                if anthropic is None:
                    st.error("anthropic SDK not installed")
                    st.stop()
                api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    st.error("Missing ANTHROPIC_API_KEY")
                    st.stop()
                client = anthropic.Anthropic(api_key=api_key)
                
                # Anthropic requires special handling for system messages
                system_text = system_content
                claude_messages = [m for m in messages_for_llm if m["role"] != "system"]
                
                # Ensure first message is from user
                if not claude_messages or claude_messages[0]["role"] != "user":
                    claude_messages.insert(0, {"role": "user", "content": "Hello"})
                
                response_placeholder = st.empty()
                full_response = ""
                
                with client.messages.stream(
                    model=model_name,
                    system=system_text,
                    messages=claude_messages,
                    max_tokens=2000,
                    temperature=0.7
                ) as stream:
                    for chunk in stream.text_stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response)
                
                response = full_response
                
            else:  # gemini
                if genai is None:
                    st.error("google-generativeai SDK not installed")
                    st.stop()
                api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    st.error("Missing GOOGLE_API_KEY")
                    st.stop()
                genai.configure(api_key=api_key)
                
                # Create model with selected model name
                model = genai.GenerativeModel(model_name)
                
                # Gemini uses a different message format
                prompt = f"{system_content}\n\n"
                for msg in messages_for_llm[1:]:
                    prompt += f"{msg['role'].upper()}: {msg['content']}\n"
                prompt += "\nASSISTANT: "
                
                response_placeholder = st.empty()
                full_response = ""
                
                # Try streaming first
                try:
                    for chunk in model.generate_content(prompt, stream=True):
                        if hasattr(chunk, 'text') and chunk.text:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response)
                except Exception:
                    # If streaming fails, try non-streaming
                    response_obj = model.generate_content(prompt)
                    full_response = response_obj.text
                    response_placeholder.markdown(full_response)
                
                response = full_response
        
        except Exception as e:
            response = f"Error generating response: {str(e)}"
            st.error(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})

# ===================== Footer Information =====================
st.sidebar.markdown("---")
st.sidebar.subheader("System Info")
st.sidebar.metric("Collection", CHROMA_COLLECTION)
st.sidebar.metric("Total Chunks", st.session_state.HW5_vectorDB.count())
st.sidebar.metric("Conversation Length", len(st.session_state.messages))
st.sidebar.metric("Memory Buffer", f"{BUFFER_PAIRS} pairs")

# Add description at the bottom
with st.expander("ℹ️ About this Enhanced Chatbot"):
    st.markdown("""
    ### Key Features:
    1. **Function-based Retrieval**: Uses `get_relevant_club_info()` to retrieve information
    2. **Short-term Memory**: Maintains last 5 conversation pairs for context
    3. **Flexible Integration**: Choose between system prompt or function result methods
    4. **Multi-provider Support**: Works with OpenAI, Anthropic, and Google Gemini
    
    ### How it works:
    - User query → Vector search function → Retrieve relevant docs → LLM generates answer
    - The retrieval function operates independently from the LLM prompting
    - Retrieved information can be passed to LLM in different ways
    """)