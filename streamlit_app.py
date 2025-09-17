import streamlit as st

st.set_page_config(page_title="HW Manager")

pages = {
    "HWs": [
        st.Page("HWs/HW1.py", title="HW1 — Document Q&A"),
        st.Page("HWs/HW2.py", title="HW2 — URL Summarizer (WIP)"),
        st.Page("HWs/HW3.py", title="HW3 — Chatbox"),
    ],
}

st.navigation(pages).run()
