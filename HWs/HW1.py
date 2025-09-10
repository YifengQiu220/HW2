import streamlit as st
from openai import OpenAI
import fitz  

st.set_page_config(page_title="HW1 — Document Q&A", page_icon="📄")

# ---------------- Title & Intro ----------------
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys). "
)

# ---------------- API Key  ----------------
openai_api_key = st.text_input("OpenAI API Key:", type="password")
if not openai_api_key:
    st.info("YOU MUST ENTER YOUR OpenAI API key!!!!!!", icon="🗝️")
    st.stop()

need_validate = (
    "last_key" not in st.session_state
    or st.session_state["last_key"] != openai_api_key
    or "api_valid" not in st.session_state
)
if need_validate:
    try:
        client = OpenAI(api_key=openai_api_key)
        client.models.list()  
        st.session_state["api_valid"] = True
        st.session_state["last_key"] = openai_api_key
        st.session_state["client"] = client
        st.success("✅ API key is valid! You are useing gpt-3.5-turbo")
    except Exception:
        st.session_state["api_valid"] = False
        st.session_state["last_key"] = openai_api_key
        st.error("❌ Invalid OpenAI API key or network error. Please check and try again.")
        st.stop()

client = st.session_state.get("client", OpenAI(api_key=openai_api_key))

# ---------------- PDF（PyMuPDF） ----------------
def read_pdf(uploaded_file) -> str:
    """Read text from a PDF uploaded via st.file_uploader using PyMuPDF."""
    data = uploaded_file.read()  # bytes
    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)

# ---------------- File upload & state ----------------
if "document_text" not in st.session_state:
    st.session_state["document_text"] = None
if "document_name" not in st.session_state:
    st.session_state["document_name"] = None

uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)", type=("txt", "pdf")
)


if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "txt":
        st.session_state["document_text"] = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state["document_name"] = uploaded_file.name
    elif ext == "pdf":
        with st.spinner("Reading PDF…"):
            st.session_state["document_text"] = read_pdf(uploaded_file)
        st.session_state["document_name"] = uploaded_file.name
    else:
        st.error("Unsupported file type.")
        st.session_state["document_text"] = None
        st.session_state["document_name"] = None
else:
    st.session_state["document_text"] = None
    st.session_state["document_name"] = None

# ---------------- Question input ----------------
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not bool(st.session_state["document_text"]),
)

# ---------------- Call OpenAI ----------------
if st.session_state["document_text"] and question:
    messages = [
        {
            "role": "user",
            "content": (
                f"Here's a document named {st.session_state['document_name']}:\n\n"
                f"{st.session_state['document_text']}\n\n---\n\n{question}"
            ),
        }
    ]

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    st.write_stream(stream)
