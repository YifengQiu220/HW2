import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time

# SDK imports 
try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# Configuration
@dataclass
class Config:
    MAX_CONTENT_LENGTH = 200_000
    REQUEST_TIMEOUT = 15
    DEFAULT_TEMPERATURE = 0.2
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    SUMMARY_TYPES = {
        "100 words": "Summarize the document in exactly 100 words",
        "2 paragraphs": "Summarize the document in 2 connected paragraphs", 
        "5 bullet points": "Summarize the document in 5 concise bullet points"
    }
    
    LANGUAGES = ["English", "Chinese", "French", "Spanish", "German", "Japanese"]
    PROVIDERS = ["OpenAI", "Anthropic", "Google Gemini"]
    
    MODEL_MAPPING = {
        "OpenAI": {
            "advanced": "gpt-5-chat-latest",
            "standard": "gpt-4o-mini",
            "key": "OPENAI_API_KEY"
        },
        "Anthropic": {
            "advanced": "claude-3-haiku-20240307",  
            "standard": "claude-3-haiku-20240307",
            "key": "ANTHROPIC_API_KEY"
        },
        "Google Gemini": {
            "advanced": "gemini-1.5-pro",
            "standard": "gemini-1.5-flash",
            "key": "GOOGLE_API_KEY"
        }
    }


# Abstract base class for LLM providers
class LLMProvider(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, model: str, temperature: float) -> str:
        pass


class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, model: str, temperature: float) -> str:
        if OpenAIClient is None:
            raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements.txt")
        
        client = OpenAIClient(api_key=self.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    def generate(self, prompt: str, model: str, temperature: float) -> str:
        if anthropic is None:
            raise RuntimeError("Anthropic SDK not installed. Add 'anthropic' to requirements.txt")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1500,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in message.content)


class GoogleGeminiProvider(LLMProvider):
    def generate(self, prompt: str, model: str, temperature: float) -> str:
        if genai is None:
            raise RuntimeError("Google Generative AI SDK not installed. Add 'google-generativeai' to requirements.txt")
        
        genai.configure(api_key=self.api_key)
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        return response.text or ""


class URLContentExtractor:
    """Handles fetching and cleaning web page content"""
    
    @staticmethod
    def extract(url: str) -> Optional[str]:
        """Fetch and clean text from a web page"""
        try:
            headers = {"User-Agent": Config.USER_AGENT}
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove noise elements
            for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header"]):
                tag.decompose()
            
            # Try to extract main content areas first
            main_content = (
                soup.find("article") or 
                soup.find("main") or 
                soup.find("div", class_="content") or
                soup.find("div", {"id": "content"}) or
                soup.body
            )
            
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)
            
            # Truncate if too long
            return text[:Config.MAX_CONTENT_LENGTH]
            
        except requests.RequestException as e:
            st.error(f"Error reading {url}: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error processing {url}: {e}")
            return None


class SummaryGenerator:
    """Handles summary generation logic"""
    
    @staticmethod
    def get_api_key(key_name: str) -> str:
        """Unified API key retrieval"""
        api_key = st.secrets.get(key_name) or os.getenv(key_name)
        if not api_key:
            raise RuntimeError(f"Missing {key_name}. Please add it to your secrets or environment variables.")
        return api_key
    
    @staticmethod
    def get_provider(provider_name: str) -> LLMProvider:
        """Factory method to create appropriate provider instance"""
        config = Config.MODEL_MAPPING.get(provider_name)
        if not config:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        api_key = SummaryGenerator.get_api_key(config["key"])
        
        if provider_name == "OpenAI":
            return OpenAIProvider(api_key)
        elif provider_name == "Anthropic":
            return AnthropicProvider(api_key)
        elif provider_name == "Google Gemini":
            return GoogleGeminiProvider(api_key)
        else:
            raise ValueError(f"Provider {provider_name} not implemented")
    
    @staticmethod
    def create_prompt(content: str, summary_type: str, language: str) -> str:
        """Create an optimized prompt for summary generation"""
        instruction = Config.SUMMARY_TYPES.get(summary_type, summary_type)
        
        prompt = f"""You are an expert content summarizer. Your task is to create a clear, accurate, and well-structured summary.

Instructions:
1. {instruction}
2. Write the summary in {language} language only
3. Preserve key facts, figures, and main arguments
4. Maintain objectivity and accuracy
5. Use clear and concise language
6. Ensure logical flow between points or paragraphs

Content to summarize:
---
{content}
---

Summary:"""
        return prompt
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def generate_summary(_provider: LLMProvider, model: str, prompt: str) -> str:
        """Generate summary with caching"""
        return _provider.generate(prompt, model, Config.DEFAULT_TEMPERATURE)


# Streamlit UI
def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="URL Summarizer",
        page_icon="📄",
        layout="wide"
    )
    st.title("🔗 Advanced URL Summarizer")
    st.markdown("Extract and summarize content from any webpage using state-of-the-art LLMs")


def setup_sidebar() -> Dict[str, Any]:
    """Setup sidebar controls and return configuration"""
    st.sidebar.header("⚙️ Configuration")
    
    with st.sidebar.expander("Summary Settings", expanded=True):
        summary_type = st.selectbox(
            "Summary Format",
            options=list(Config.SUMMARY_TYPES.keys()),
            help="Choose how you want the content to be summarized"
        )
        
        language = st.selectbox(
            "Output Language",
            options=Config.LANGUAGES,
            help="Language for the generated summary"
        )
    
    with st.sidebar.expander("Model Settings", expanded=True):
        provider = st.selectbox(
            "LLM Provider",
            options=Config.PROVIDERS,
            help="Select which AI provider to use"
        )
        
        use_advanced = st.checkbox(
            "Use Advanced Model",
            value=False,
            help="Advanced models provide better quality but may be slower"
        )
    
    # Display selected model info
    model_config = Config.MODEL_MAPPING[provider]
    model_name = model_config["advanced"] if use_advanced else model_config["standard"]
    
    
    return {
        "summary_type": summary_type,
        "language": language,
        "provider": provider,
        "use_advanced": use_advanced,
        "model_name": model_name
    }


def main():
    """Main application logic"""
    setup_page()
    config = setup_sidebar()
    
    # URL input
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input(
            "Enter URL to summarize:",
            placeholder="https://example.com/article",
            help="Paste any article or webpage URL"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        summarize_button = st.button("🚀 Summarize", type="primary", use_container_width=True)
    
    # Process URL when provided
    if url and summarize_button:
        try:
            # Extract content
            with st.spinner("📖 Reading webpage content..."):
                content = URLContentExtractor.extract(url)
            
            if not content:
                st.error("Failed to extract content from the URL. Please check the URL and try again.")
                return
            
            # Show content preview
            with st.expander("📄 Extracted Content Preview", expanded=False):
                st.text(content[:1000] + "..." if len(content) > 1000 else content)
                st.caption(f"Total length: {len(content)} characters")
            
            # Generate summary
            with st.spinner(f"🤖 Generating summary using {config['provider']}..."):
                start_time = time.time()
                
                # Create provider and generate
                provider = SummaryGenerator.get_provider(config["provider"])
                prompt = SummaryGenerator.create_prompt(
                    content, 
                    config["summary_type"],
                    config["language"]
                )
                summary = SummaryGenerator.generate_summary(
                    provider,
                    config["model_name"],
                    prompt
                )
                
                elapsed_time = time.time() - start_time
            
            # Display results
            st.success(f"✅ Summary generated successfully in {elapsed_time:.1f} seconds!")
            
            st.subheader("📝 Generated Summary")
            st.markdown(summary)
            
            # Add copy button
            st.button(
                "📋 Copy to Clipboard",
                on_click=lambda: st.write(summary),
                help="Click to copy the summary"
            )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Provider", config["provider"])
            with col2:
                st.metric("Model Type", "Advanced" if config["use_advanced"] else "Standard")
            with col3:
                st.metric("Generation Time", f"{elapsed_time:.1f}s")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your API keys and try again.")


if __name__ == "__main__":
    main()