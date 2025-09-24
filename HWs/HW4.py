# HW4.py — Persistent ChromaDB with HTML files (RAG)

import os
import glob
import textwrap
from typing import List, Dict

import streamlit as st
from openai import OpenAI

# Optional providers (guarded imports so app still runs if libs/keys are missing)
try:
    import anthropic
except Exception:
    anthropic = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


# ---- SQLite fix (Codespaces/Streamlit Cloud often need this)
try:
    __import__("pysqlite3")  # noqa: F401
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup

# ===================== Constants =====================
PAGE_TITLE        = "HW4 – iSchool chatbot using RAG"
ORGS_FOLDER       = os.path.join("HWs", "hw4_orgs")    # folder containing unzipped HTML files
CHROMA_PATH       = "./ChromaDB_hw4"                   # persistent Chroma directory
CHROMA_COLLECTION = "HW4Collection"                    # separate name for HW4
EMBED_MODEL       = "text-embedding-3-small"           # OpenAI embeddings (1536-dim)
TOP_K             = 3
BUFFER_PAIRS      = 5
MODEL_NAME        = "gpt-4o-mini"

# ===================== UI Header =====================
st.title(PAGE_TITLE)
st.caption(
    "This app builds a persistent ChromaDB from the provided HTML files and uses "
    "retrieval-augmented generation (RAG) to answer questions."
)

# ===================== Chunking (exactly 2 per document) =====================
def chunk_into_two(text: str) -> List[str]:
    """
    We split EACH HTML document into EXACTLY TWO mini-documents (chunks).

    Method (explained): a length-based 50/50 split with a soft boundary.
    - We find the midpoint by length, then try to snap to a sentence boundary
      near the midpoint to avoid cutting in the middle of a sentence.
    - If we can't find a period near the midpoint, we split exactly at the midpoint.

    Why: satisfies the HW requirement ("two separate mini-documents per doc"),
    keeps chunks coherent, and avoids sending an entire page as one chunk.
    """
    text = " ".join(text.split())
    if not text:
        return ["", ""]
    if len(text) <= 400:  # very short pages → just split in half
        mid = len(text) // 2
        return [text[:mid].strip(), text[mid:].strip()]

    mid = len(text) // 2
    # Search for a period+space near midpoint (±200 chars) to reduce sentence cuts
    left = max(0, mid - 200)
    right = min(len(text), mid + 200)
    window = text[left:right]
    snap = window.rfind(". ")
    if snap != -1:
        cut = left + snap + 1  # cut right after the period
    else:
        cut = mid

    return [text[:cut].strip(), text[cut:].strip()]

# ===================== Build ChromaDB (persistent, only once) =====================
@st.cache_resource(show_spinner=True)
def build_chromadb() -> chromadb.Collection:
    """
    Create or load the persistent Chroma collection with OpenAI embeddings.
    We only add documents when the collection is empty (build once).
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to .streamlit/secrets.toml or environment.")
        st.stop()

    # Ensure source folder exists and has HTML files
    if not os.path.isdir(ORGS_FOLDER):
        st.error(f"Source folder not found: {os.path.abspath(ORGS_FOLDER)}")
        st.info("Create the folder and copy all HTML files from su_orgs.zip, then rerun.")
        st.stop()

    html_paths = sorted(glob.glob(os.path.join(ORGS_FOLDER, "*.html")))
    if not html_paths:
        st.error(f"No HTML files found in {os.path.abspath(ORGS_FOLDER)}")
        st.info("Copy the unzipped HTML pages into this folder and rerun.")
        st.stop()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=EMBED_MODEL,
    )

    # Use get_or_create with embedding_function; if the collection already exists with
    # a different EF this would normally error — in that case delete the directory
    # CHROMA_PATH/HW4Collection and rebuild.
    coll = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
    )

    # Only build if empty (create-once behavior)
    if coll.count() == 0:
        ids, docs, metas = [], [], []
        for p in html_paths:
            fname = os.path.basename(p)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                # Extract visible text from HTML
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
            # Add in batches to avoid large single calls
            batch = 100
            for i in range(0, len(ids), batch):
                coll.add(
                    documents=docs[i:i+batch],
                    metadatas=metas[i:i+batch],
                    ids=ids[i:i+batch],
                )
            st.sidebar.success(f"Indexed {len(ids)} chunks from {len(html_paths)} HTML files")

    return coll

# Build (or load) the collection once per run and keep a handle
if "HW4_vectorDB" not in st.session_state:
    st.session_state.HW4_vectorDB = build_chromadb()

# ===================== Retrieval =====================
def retrieve_context(query: str, k: int = TOP_K) -> str:
    """Return a compact context block from top-k retrieved chunks."""
    try:
        coll = st.session_state.HW4_vectorDB
        res = coll.query(query_texts=[query], n_results=k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        parts = []
        for d, m in zip(docs, metas):
            if not d:
                continue
            fname = (m or {}).get("filename", "unknown.html")
            snippet = textwrap.shorten(d.replace("\n", " "), width=480, placeholder="…")
            parts.append(f"[{fname}] {snippet}")

        if not parts:
            return ""
        return (
            "RETRIEVED CONTEXT (from HTML):\n"
            + "\n".join(parts)
            + "\n\nUse the above context if helpful. Mention filenames when used."
        )
    except Exception as e:
        return f"(Retrieval error: {e})"

# ===================== OpenAI client =====================
@st.cache_resource
def get_openai_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

@st.cache_resource
def get_llm_client(provider_name: str, model_name: str):
    """Return an initialized client/handle for the chosen provider."""
    if provider_name == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Missing OPENAI_API_KEY")
            st.stop()
        return ("openai", OpenAI(api_key=api_key))
    elif provider_name == "Anthropic":
        if anthropic is None:
            st.error("anthropic SDK not installed. Add `anthropic` to requirements.txt")
            st.stop()
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("Missing ANTHROPIC_API_KEY")
            st.stop()
        return ("anthropic", anthropic.Anthropic(api_key=api_key))
    else:  # Google (Gemini)
        if genai is None:
            st.error("google-generativeai SDK not installed. Add `google-generativeai`")
            st.stop()
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Missing GOOGLE_API_KEY")
            st.stop()
        genai.configure(api_key=api_key)
        # For Gemini we return the model object itself
        return ("gemini", genai.GenerativeModel(model_name))


# ---- Provider & Model choice (3 different LLMs)
st.sidebar.header("Model")
provider = st.sidebar.selectbox("Provider", ["OpenAI", "Anthropic", "Google (Gemini)"])
tier = st.sidebar.radio("Cost tier", ["Cheap", "Expensive"], horizontal=True)

# Map provider+tier -> concrete model name
if provider == "OpenAI":
    selected_model = "gpt-4o-mini" if tier == "Cheap" else "gpt-4o"
elif provider == "Anthropic":
    # Use Haiku for both if you don’t have Sonnet access
    selected_model = "claude-3-haiku-20240307" if tier == "Cheap" else "claude-3-haiku-20240307"
else:  # Google
    selected_model = "gemini-1.5-flash" if tier == "Cheap" else "gemini-1.5-pro"

# ===================== Chat state =====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! Ask me about Syracuse iSchool student organizations."}
    ]

# st.sidebar.header("Options")
# buffer_pairs = st.sidebar.slider("Buffer size (exchanges)", 1, 10, BUFFER_PAIRS)
buffer_pairs = 5
if st.sidebar.button("Clear Chat 🗑️"):
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! Ask me about Syracuse iSchool student organizations."}
    ]
    st.rerun()

def get_buffered(messages: List[Dict[str, str]], pairs: int) -> List[Dict[str, str]]:
    """Keep the welcome + last N user/assistant turns."""
    if len(messages) <= 2:
        return messages
    head = [messages[0]]
    tail = messages[-(pairs * 2):]
    return head + tail

# ===================== Provider / Model Selection =====================
st.sidebar.header("Model")

provider_kind = st.sidebar.selectbox(
    "Provider",
    ["openai", "anthropic", "gemini"],
    format_func=lambda x: x.title()
)

if provider_kind == "openai":
    selected_model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )
    from openai import OpenAI
    llm_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

elif provider_kind == "anthropic":
    selected_model = st.sidebar.selectbox(
        "Anthropic Model",
        ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    )
    import anthropic
    llm_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

else:  # gemini
    selected_model = st.sidebar.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"]
    )
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    llm_client = genai.GenerativeModel(selected_model)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input + RAG
if user_text := st.chat_input("Ask about student orgs…"):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    rag_context = retrieve_context(user_text, k=TOP_K)

    sys_blocks = [
        "You are a helpful TA for Syracuse iSchool. Answer clearly and concisely for students.",
        "When your answer uses retrieved context, explicitly cite the filenames.",
        "If you don't know the answer say 'I don't know'."
    ]
    if rag_context:
        sys_blocks.append(rag_context)

    system_msg = {"role": "system", "content": "\n\n".join(sys_blocks)}
    msgs = [system_msg] + get_buffered(st.session_state.messages, buffer_pairs)

    with st.chat_message("assistant"):
        st.caption(f"Provider: {provider_kind.title()} • Model: {selected_model} • Retrieval: ChromaDB")

        try:
            if provider_kind == "openai":
                # OpenAI: native streaming
                stream = llm_client.chat.completions.create(
                    model=selected_model,
                    messages=msgs,
                    temperature=0.7,
                    stream=True,
                )
                answer = st.write_stream(stream)

            elif provider_kind == "anthropic":
                # Anthropic: system 独立，消息只保留 user/assistant
                system_text = ""
                claude_msgs = []
                for m in msgs:
                    if m["role"] == "system":
                        system_text = f"{system_text}\n\n{m['content']}" if system_text else m["content"]
                    elif m["role"] in ("user", "assistant"):
                        claude_msgs.append({"role": m["role"], "content": m["content"]})
                if not claude_msgs or claude_msgs[0]["role"] != "user":
                    claude_msgs.insert(0, {"role": "user", "content": "Hello"})

                buf = ""
                out = st.empty()
                with llm_client.messages.stream(
                    model=selected_model,
                    system=system_text or None,
                    messages=claude_msgs,
                    max_tokens=2000,
                    temperature=0.7,
                ) as stream:
                    for chunk in stream.text_stream:
                        buf += chunk
                        out.markdown(buf)
                answer = buf

            else:  # provider_kind == "gemini"
                # Gemini: build single long prompt
                full_prompt = ""
                for m in msgs:
                    if m["role"] == "system":
                        full_prompt += f"SYSTEM:\n{m['content']}\n\n"
                full_prompt += "CONVERSATION:\n"
                for m in msgs:
                    if m["role"] != "system":
                        full_prompt += f"{m['role'].upper()}: {m['content']}\n"
                full_prompt += "\nASSISTANT:\n"

                buf = ""
                out = st.empty()
                for chunk in llm_client.generate_content(full_prompt, stream=True):
                    if getattr(chunk, "text", None):
                        buf += chunk.text
                        out.markdown(buf)
                answer = buf

        except Exception as e:
            answer = f"Error calling {provider_kind.title()}: {e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})



# Footer
st.sidebar.markdown("---")
st.sidebar.metric("Collection", CHROMA_COLLECTION)
st.sidebar.metric("Indexed chunks", st.session_state.HW4_vectorDB.count())
