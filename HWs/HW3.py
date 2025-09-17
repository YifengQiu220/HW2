import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import anthropic
import google.generativeai as genai
import tiktoken

# ---------------------- Page & Sidebar ----------------------
st.title("HW3 — Streaming Chatbot with URL Context & Memory")

st.sidebar.header("Configuration")

# URLs (two inputs) - Requirement 2
st.sidebar.markdown("### URL Sources")
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/article-1")
url2 = st.sidebar.text_input("URL 2 (Optional)", placeholder="https://example.com/article-2")

# LLM Provider selection - Requirement 3
st.sidebar.markdown("### LLM Selection")
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OpenAI", "Anthropic", "Google Gemini"]
)

# Model selection based on provider - Requirement 3a,b
if provider == "OpenAI":
    model_type = st.sidebar.radio(
        "Model Type",
        ["Cheap", "Expensive"]
    )
    model_to_use = "gpt-4o-mini" if model_type == "Cheap" else "gpt-4o"
elif provider == "Anthropic":
    model_type = st.sidebar.radio(
        "Model Type",
        ["Cheap", "Expensive"]
    )
    model_to_use = "claude-3-haiku-20240307" if model_type == "Cheap" else "claude-3-haiku-20240307"
else:  # Google Gemini
    model_type = st.sidebar.radio(
        "Model Type",
        ["Cheap", "Expensive"]
    )
    model_to_use = "gemini-1.5-flash" if model_type == "Cheap" else "gemini-1.5-pro"

# Memory type selection - Requirement 4
st.sidebar.markdown("### Conversation Memory")
memory_type = st.sidebar.selectbox(
    "Memory Type",
    ["Buffer (6 questions)", "Conversation Summary", "Token Buffer (2000 tokens)"]
)

# ---------------------- Initialize Clients ----------------------
@st.cache_resource
def get_client(provider_name):
    """Initialize and return the appropriate client based on provider"""
    if provider_name == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY in secrets")
            return None
        return OpenAI(api_key=api_key)
    
    elif provider_name == "Anthropic":
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Missing ANTHROPIC_API_KEY in secrets")
            return None
        return anthropic.Anthropic(api_key=api_key)
    
    else:  # Google Gemini
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Missing GOOGLE_API_KEY in secrets")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_to_use)

# ---------------------- Helpers ----------------------
@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(messages):
    """Approx token count for messages."""
    encoding = get_tokenizer()
    total = 0
    for m in messages:
        total += len(encoding.encode(m["role"]))
        total += len(encoding.encode(m["content"]))
        total += 4
    total += 3
    return total

def get_message_buffered(messages, keep_pairs):
    """Keep last N user-assistant exchanges (+ keep system messages)."""
    if len(messages) <= 1:
        return messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    to_keep = keep_pairs * 2
    if len(other) > to_keep:
        other = other[-to_keep:]
    return system_msgs + other

def get_token_buffered(messages, max_tok):
    """Keep as many messages as fit under max_tok (keep system messages)."""
    if not messages:
        return []
    system_msgs = [m for m in messages if m["role"] == "system"]
    other = [m for m in messages if m["role"] != "system"]
    res = []
    total = count_tokens(system_msgs)
    for m in reversed(other):
        t = count_tokens([m])
        if total + t <= max_tok:
            res.insert(0, m)
            total += t
        else:
            break
    return system_msgs + res

@st.cache_data(show_spinner=False)
def read_url_content(url: str) -> str | None:
    """Extract content from URL - Requirement 6"""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        text = "\n".join([ln for ln in lines if ln])
        return text[:10000]
    except Exception as e:
        st.sidebar.error(f"Read URL failed: {e}")
        return None

def update_summary(old_summary: str, last_turn: list, provider_name: str) -> str:
    """Update running conversation summary."""
    try:
        prompt = f"""Update the running summary of the conversation for later recall.
OLD SUMMARY:
{old_summary}

NEW TURN:
User: {last_turn[0]["content"]}
Assistant: {last_turn[1]["content"]}

Return a concise updated summary (no more than 120 words)."""
        
        if provider_name == "OpenAI":
            client = get_client("OpenAI")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        else:
            return old_summary + "\n" + f"User asked about: {last_turn[0]['content'][:50]}..."
    except Exception:
        return old_summary

# ---------------------- Session State ----------------------
if "conv_summary" not in st.session_state:
    st.session_state.conv_summary = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help you discuss content from URLs. Please provide 1-2 URLs in the sidebar."}
    ]

if "url_context" not in st.session_state:
    st.session_state.url_context = ""

# Clear conversation
if st.sidebar.button("Clear Conversation 🗑️"):
    st.session_state.conv_summary = ""
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help you discuss content from URLs. Please provide 1-2 URLs in the sidebar."}
    ]
    st.rerun()

# ---------------------- URL Loading Status ----------------------
url1_text = read_url_content(url1) if url1 else None
url2_text = read_url_content(url2) if url2 else None

# Build and store URL context
if url1_text or url2_text:
    combined = "You have access to the following web content. Use it to answer questions:\n\n"
    if url1_text:
        combined += f"=== Content from URL 1: {url1} ===\n"
        combined += f"{url1_text}\n\n"
    if url2_text:
        combined += f"=== Content from URL 2: {url2} ===\n"
        combined += f"{url2_text}\n\n"
    combined += "Please use this content to answer the user's questions. Cite which URL source you're using when relevant."
    st.session_state.url_context = combined
else:
    st.session_state.url_context = ""

ready = []
if url1_text: ready.append("URL 1")
if url2_text: ready.append("URL 2")
if ready:
    st.sidebar.success(f"✅ Loaded: {', '.join(ready)}")

# ---------------------- Display History ----------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ---------------------- Generate Streaming Response - Requirement 7 ----------------------
def generate_streaming_response(messages, provider_name, model):
    """Generate streaming response based on provider"""
    client = get_client(provider_name)
    if not client:
        return "Error: Client not initialized. Please check API keys."
    
    if provider_name == "OpenAI":
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        return st.write_stream(stream)
    
    elif provider_name == "Anthropic":
        system_content = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                if system_content:
                    system_content += "\n\n" + msg["content"]
                else:
                    system_content = msg["content"]
            elif msg["role"] in ["user", "assistant"]:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
        
        if not claude_messages:
            claude_messages = [{"role": "user", "content": "Hello"}]
        elif claude_messages[0]["role"] != "user":
            claude_messages.insert(0, {"role": "user", "content": "Hello"})
        
        try:
            with client.messages.stream(
                model=model,
                messages=claude_messages,
                system=system_content if system_content else None,
                max_tokens=2000,
                temperature=0.7
            ) as stream:
                response = ""
                response_container = st.empty()
                for text in stream.text_stream:
                    response += text
                    response_container.markdown(response)
                return response
        except Exception as e:
            return f"Anthropic API error: {str(e)}"
    
    else:
        try:
            full_prompt = ""
            
            # Add URL context explicitly for Gemini
            if st.session_state.url_context:
                full_prompt = "IMPORTANT CONTEXT FROM URLS:\n" + st.session_state.url_context + "\n\n"
            
            # Add conversation history
            full_prompt += "CONVERSATION:\n"
            for msg in messages:
                if msg["role"] != "system":
                    full_prompt += f"{msg['role'].upper()}: {msg['content']}\n"
            
            full_prompt += "\nASSISTANT: Based on the URL content provided above, I will answer your question.\n"
            
            response = client.generate_content(full_prompt, stream=True)
            
            full_response = ""
            response_container = st.empty()
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    response_container.markdown(full_response)
            return full_response
        except Exception as e:
            return f"Google API error: {str(e)}"

# ---------------------- Chat Input & Generation ----------------------
if user_text := st.chat_input("Ask about the URLs..."):
    if not st.session_state.url_context:
        st.warning("Please provide at least one URL in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Build messages for API
        context_msgs = []
        if st.session_state.url_context:
            context_msgs.append({
                "role": "system",
                "content": st.session_state.url_context
            })

        # Apply memory strategy
        if memory_type == "Buffer (6 questions)":
            messages_for_api = get_message_buffered(st.session_state.messages, keep_pairs=6)
        elif memory_type == "Token Buffer (2000 tokens)":
            messages_for_api = get_token_buffered(st.session_state.messages, max_tok=2000)
        else:
            trimmed = get_message_buffered(st.session_state.messages, keep_pairs=2)
            if st.session_state.conv_summary:
                summary_msg = {
                    "role": "system",
                    "content": "Conversation summary so far:\n" + st.session_state.conv_summary
                }
                messages_for_api = [summary_msg] + trimmed
            else:
                messages_for_api = trimmed

        # Prepend URL context
        messages_for_api = context_msgs + messages_for_api

        # Generate streaming response
        with st.chat_message("assistant"):
            st.caption(f"Using {provider} - {model_to_use}")
            try:
                assistant_text = generate_streaming_response(messages_for_api, provider, model_to_use)
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                
                if memory_type == "Conversation Summary" and len(st.session_state.messages) >= 2:
                    last_turn = st.session_state.messages[-2:]
                    st.session_state.conv_summary = update_summary(
                        st.session_state.conv_summary, last_turn, provider
                    )
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ---------------------- Sidebar Metrics ----------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")

if memory_type == "Buffer (6 questions)":
    _buf_preview = get_message_buffered(st.session_state.messages, keep_pairs=6)
elif memory_type == "Token Buffer (2000 tokens)":
    _buf_preview = get_token_buffered(st.session_state.messages, max_tok=2000)
else:
    _buf_preview = get_message_buffered(st.session_state.messages, keep_pairs=2)

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Messages", len(st.session_state.messages))
with col2:
    st.metric("In Memory", len(_buf_preview))

# Display current configuration
st.markdown("---")
info_text = f"""
**Current Configuration:**
- Provider: {provider} ({model_type})
- Memory: {memory_type}
- URLs Loaded: {len(ready)}/2
"""
st.info(info_text)

# Show URL content preview if loaded
if st.session_state.url_context:
    with st.expander("📄 View Loaded URL Content"):
        st.text(st.session_state.url_context[:1000] + "..." if len(st.session_state.url_context) > 1000 else st.session_state.url_context)