import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import google.generativeai as genai
import json
from datetime import datetime

# Import new langchain versions
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    st.error("Please install: pip install langchain-chroma langchain-openai")
    st.stop()

from langchain.schema import Document

# Initialize
st.title("HW7 - News Info Bot for Global Law Firm")

# Check API keys
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Google Gemini client (optional)
google_client = None
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    google_client = genai

CSV_PATH = os.path.join("HWs", "HW7data.csv")
VECTOR_DB_PATH = os.path.join("ChromaDB_HW7")

# --- Step 1: Load CSV ---
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return None

df = load_data()
if df is None:
    st.error("CSV not found. Please check the path.")
    st.stop()

st.success("CSV loaded successfully!")
with st.expander("View raw data"):
    st.dataframe(df.head(10))

# --- Step 2: Create / Load VectorDB ---
@st.cache_resource
def build_or_load_vectordb(_df):
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        st.info("Loading existing ChromaDB...")
        try:
            db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            return db
        except Exception as e:
            st.warning(f"Failed to load existing DB: {e}. Creating new one...")
            import shutil
            shutil.rmtree(VECTOR_DB_PATH)
    
    st.info("Creating new ChromaDB...")
    docs = []
    for idx, row in _df.iterrows():
        content = f"""
Company: {row['company_name']}
Date: {row['Date']}
News Content: {row['Document']}
        """.strip()
        
        metadata = {
            "id": str(idx),
            "company": str(row['company_name']),
            "date": str(row['Date'])
        }
        docs.append(Document(page_content=content, metadata=metadata))
    
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    
    st.success("VectorDB created and saved!")
    return db

db = build_or_load_vectordb(df)

# --- Session State for Test Results ---
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Vendor Selection
    vendor = st.selectbox(
        "Choose Vendor:",
        ["OpenAI", "Google Gemini"],
        help="Note: If Gemini has issues, use Comparison Mode with OpenAI models only"
    )
    
    # Model Selection based on vendor
    if vendor == "OpenAI":
        model_options = {
            "GPT-4o (Expensive)": "gpt-4o",
            "GPT-4o-mini (Cheap)": "gpt-4o-mini",
            "GPT-3.5-turbo (Cheapest)": "gpt-3.5-turbo"
        }
    else:
        if google_client is None:
            st.warning("Google API key not found")
            model_options = {}
        else:
            model_options = {
                "Gemini 2.5 Pro (Expensive)": "models/gemini-2.5-pro-preview-03-25",
                "Gemini 2.5 Flash (Cheap)": "models/gemini-2.5-flash",
                "Gemini 2.5 Flash Lite (Cheapest)": "models/gemini-2.5-flash-lite-preview-06-17"
            }
    
    if model_options:
        model_display = st.selectbox("Choose Model:", list(model_options.keys()))
        model_id = model_options[model_display]
    else:
        st.error("No models available")
        st.stop()
    
    st.markdown("---")
    st.write(f"**Current Setup:**")
    st.write(f"- Vendor: {vendor}")
    st.write(f"- Model: {model_display}")
    st.write(f"- Total News: {len(df)}")
    
    st.markdown("---")
    
    # Comparison Mode
    compare_mode = st.checkbox("Comparison Mode", help="Test with multiple models simultaneously")
    
    if compare_mode:
        st.info("In comparison mode, your query will be sent to multiple models")
    
    st.markdown("---")
    
    if st.button("Rebuild Vector DB"):
        import shutil
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("View Test History"):
        if st.session_state.test_results:
            st.dataframe(pd.DataFrame(st.session_state.test_results))
        else:
            st.info("No test results yet")
    
    if st.button("Clear History"):
        st.session_state.test_results = []
        st.success("History cleared!")
    
    if st.button("Download Test Results"):
        if st.session_state.test_results:
            results_df = pd.DataFrame(st.session_state.test_results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="test_results.csv",
                mime="text/csv"
            )

# --- Define Function Tools ---
functions_openai = [
    {
        "name": "rank_news_by_interest",
        "description": "Rank news articles by relevance to a global law firm. Consider legal risk, regulatory changes, litigation, M&A, compliance issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "ranking_criteria": {
                    "type": "string",
                    "description": "Criteria for ranking (e.g., 'legal_risk', 'regulatory_impact')"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top news items to return",
                    "default": 5
                }
            },
            "required": ["ranking_criteria"]
        }
    },
    {
        "name": "search_news_by_topic",
        "description": "Search for news articles related to a specific topic, company, or legal area",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic, company name, or legal area to search for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    }
]

# Gemini function declarations
functions_gemini = [
    {
        "name": "rank_news_by_interest",
        "description": "Rank news articles by relevance to a global law firm. Consider legal risk, regulatory changes, litigation, M&A, compliance issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "ranking_criteria": {
                    "type": "string",
                    "description": "Criteria for ranking (e.g., 'legal_risk', 'regulatory_impact')"
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top news items to return"
                }
            },
            "required": ["ranking_criteria"]
        }
    },
    {
        "name": "search_news_by_topic",
        "description": "Search for news articles related to a specific topic, company, or legal area",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic, company name, or legal area to search for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results"
                }
            },
            "required": ["topic"]
        }
    }
]

# --- Helper Function: Call LLM ---
def call_llm(vendor, model_id, prompt, use_functions=True):
    """Unified LLM calling interface"""
    if vendor == "OpenAI":
        messages = [
            {"role": "system", "content": "You are a legal news analyst for a global law firm."},
            {"role": "user", "content": prompt}
        ]
        
        if use_functions:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in functions_openai],
                tool_choice="auto",
                temperature=0.2
            )
        else:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.3
            )
        return response
    
    elif vendor == "Google Gemini":
        try:
            # Try to list available models to help debug
            try:
                available_models = [m.name for m in genai.list_models()]
                if not any(model_id in m for m in available_models):
                    st.warning(f"Model {model_id} may not be available. Available models: {available_models[:5]}")
            except:
                pass
            
            model = genai.GenerativeModel(model_name=model_id)
            
            # Gemini: use simpler approach without function calling
            full_prompt = f"""You are a legal news analyst for a global law firm.

User query: {prompt}

Please analyze this query and provide a helpful response."""
            
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )
            return response
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                st.error(f"Gemini model '{model_id}' not found. Try: gemini-pro or gemini-1.5-pro-latest")
            else:
                st.error(f"Gemini API error: {error_msg}")
            return None
    
    return None

# --- Helper Function: Extract Function Call ---
def extract_function_call(response, vendor):
    """Extract function call information from response"""
    if vendor == "OpenAI":
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            return {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
    
    elif vendor == "Google Gemini":
        # For Gemini, we'll parse the query intent directly
        # Since Gemini doesn't use function calling, return None
        # and handle it differently in the main logic
        return None
    
    return None

# --- Helper Function: Get Final Response ---
def get_final_response(response, vendor):
    """Extract final text response"""
    if vendor == "OpenAI":
        return response.choices[0].message.content
    
    elif vendor == "Google Gemini":
        if response and hasattr(response, 'text'):
            return response.text
        elif response and hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            return part.text
    
    return "No response generated"

# --- Main Query Interface ---
st.markdown("---")
st.subheader("Ask Questions")

# Example queries
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Most Interesting News"):
        query = "find the most interesting news for a law firm"
    else:
        query = None

with col2:
    if st.button("Search Tesla"):
        query = "find news about Tesla"
    else:
        query = query if 'query' in locals() else None

with col3:
    if st.button("Regulatory News"):
        query = "find news about regulatory changes"
    else:
        query = query if 'query' in locals() else None

query_input = st.text_input(
    "Or enter your own query:",
    placeholder="e.g., 'find the most interesting news' or 'find news about regulatory changes'"
)

if query_input:
    query = query_input

if query:
    # Determine which models to test
    if compare_mode:
        test_configs = [
            ("OpenAI", "gpt-4o", "GPT-4o (Expensive)"),
            ("OpenAI", "gpt-4o-mini", "GPT-4o-mini (Cheap)")
        ]
        if google_client:
            test_configs.append(("Google Gemini", "models/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro (Expensive)"))
            test_configs.append(("Google Gemini", "models/gemini-2.5-flash", "Gemini 2.5 Flash (Cheap)"))
    else:
        test_configs = [(vendor, model_id, model_display)]
    
    # Process each model
    for test_vendor, test_model, test_display in test_configs:
        with st.spinner(f"Processing with {test_display}..."):
            try:
                # For Google Gemini, use a different approach (no function calling)
                if test_vendor == "Google Gemini":
                    # Detect query intent
                    query_lower = query.lower()
                    
                    if "interesting" in query_lower or "important" in query_lower or "rank" in query_lower:
                        # Rank news
                        all_results = db.similarity_search("legal risk regulatory compliance litigation", k=20)
                        context = "\n\n---\n\n".join([f"[{i+1}] {r.page_content}" for i, r in enumerate(all_results)])
                        
                        final_prompt = f"""
You are analyzing news for a global law firm. Rank these news articles by legal relevance.

Consider: Legal/regulatory risk, potential litigation, client impact, compliance issues, market disruption.

News articles:
{context}

Provide a ranked list of the top 5 most relevant items.
Format: 
1. [Company] - [Brief headline]
   Reason: [Why this matters to a law firm]
                        """
                        
                        st.info(f"**{test_display}** analyzing and ranking news...")
                    
                    else:
                        # Search by topic
                        # Extract topic from query
                        import re
                        topic_match = re.search(r'about (.+?)(?:\?|$)', query_lower)
                        if topic_match:
                            topic = topic_match.group(1).strip()
                        else:
                            # Just use the whole query
                            topic = query
                        
                        results = db.similarity_search(topic, k=5)
                        context = "\n\n---\n\n".join([f"[{i+1}] {r.page_content}" for i, r in enumerate(results)])
                        
                        final_prompt = f"""
Here are news articles related to "{topic}":

{context}

Provide a summary and explain relevance to a global law firm.
                        """
                        
                        st.info(f"**{test_display}** searching for: {topic}")
                    
                    # Get response from Gemini
                    final_response = call_llm(
                        test_vendor,
                        test_model,
                        final_prompt,
                        use_functions=False
                    )
                    
                    if final_response:
                        final_text = get_final_response(final_response, test_vendor)
                        
                        # Display results
                        st.markdown(f"### Results from **{test_display}**:")
                        st.markdown(final_text)
                        
                        # Save to history
                        st.session_state.test_results.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'vendor': test_vendor,
                            'model': test_display,
                            'function_used': 'Intent Detection',
                            'response_length': len(final_text),
                            'response_preview': final_text[:200] + "..."
                        })
                    
                else:
                    # OpenAI with function calling
                    initial_response = call_llm(
                        test_vendor,
                        test_model,
                        query,
                        use_functions=True
                    )
                    
                    if initial_response is None:
                        st.error(f"Failed to get response from {test_display}")
                        continue
                    
                    # Extract function call
                    func_call = extract_function_call(initial_response, test_vendor)
                    
                    if func_call:
                        st.info(f"**{test_display}** using function: **{func_call['name']}**")
                        st.json(func_call['arguments'])
                        
                        # Execute function logic
                        if func_call['name'] == "rank_news_by_interest":
                            all_results = db.similarity_search("legal risk regulatory compliance litigation", k=20)
                            context = "\n\n---\n\n".join([f"[{i+1}] {r.page_content}" for i, r in enumerate(all_results)])
                            
                            final_prompt = f"""
You are analyzing news for a global law firm. Rank these news articles by {func_call['arguments'].get('ranking_criteria', 'overall relevance')}.

Consider: Legal/regulatory risk, potential litigation, client impact, compliance issues, market disruption.

News articles:
{context}

Provide a ranked list of the top {func_call['arguments'].get('top_n', 5)} items.
Format: 
1. [Company] - [Brief headline]
   Reason: [Why this matters to a law firm]
                            """
                        
                        elif func_call['name'] == "search_news_by_topic":
                            topic = func_call['arguments']['topic']
                            max_results = func_call['arguments'].get('max_results', 5)
                            
                            results = db.similarity_search(topic, k=max_results)
                            context = "\n\n---\n\n".join([f"[{i+1}] {r.page_content}" for i, r in enumerate(results)])
                            
                            final_prompt = f"""
Here are news articles related to "{topic}":

{context}

Provide a summary and explain relevance to a global law firm.
                            """
                        
                        # Get final response
                        final_response = call_llm(
                            test_vendor,
                            test_model,
                            final_prompt,
                            use_functions=False
                        )
                        
                        final_text = get_final_response(final_response, test_vendor)
                        
                        # Display results
                        st.markdown(f"### Results from **{test_display}**:")
                        st.markdown(final_text)
                        
                        # Save to history
                        st.session_state.test_results.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'vendor': test_vendor,
                            'model': test_display,
                            'function_used': func_call['name'],
                            'response_length': len(final_text),
                            'response_preview': final_text[:200] + "..."
                        })
                    
                    else:
                        # Direct response without function
                        final_text = get_final_response(initial_response, test_vendor)
                        st.markdown(f"### Response from **{test_display}**:")
                        st.markdown(final_text)
                        
                        # Save to history
                        st.session_state.test_results.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query,
                            'vendor': test_vendor,
                            'model': test_display,
                            'function_used': 'None',
                            'response_length': len(final_text),
                            'response_preview': final_text[:200] + "..."
                        })
                
                if compare_mode and len(test_configs) > 1:
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error with {test_display}: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

# --- Display Statistics ---
if st.session_state.test_results:
    st.markdown("---")
    st.subheader("Test Statistics")
    
    results_df = pd.DataFrame(st.session_state.test_results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tests", len(results_df))
    with col2:
        st.metric("Unique Queries", results_df['query'].nunique())
    with col3:
        st.metric("Models Tested", results_df['model'].nunique())
    
    # Show summary by model
    st.markdown("#### Results by Model")
    summary = results_df.groupby('model').agg({
        'query': 'count',
        'response_length': 'mean'
    }).rename(columns={'query': 'num_tests', 'response_length': 'avg_response_length'})
    st.dataframe(summary)