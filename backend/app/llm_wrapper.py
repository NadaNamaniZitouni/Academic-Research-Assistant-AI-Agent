from typing import Optional, List
import os
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

# Option 1: Local model with transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Option 2: OpenAI
OPENAI_AVAILABLE = False
OPENAI_CLASS = None
try:
    try:
        from langchain.llms import OpenAI
        OPENAI_CLASS = OpenAI
        OPENAI_AVAILABLE = True
        print("OpenAI imported from langchain.llms")
    except ImportError:
        try:
            # Try alternative import path for newer LangChain versions
            from langchain_openai import OpenAI
            OPENAI_CLASS = OpenAI
            OPENAI_AVAILABLE = True
            print("OpenAI imported from langchain_openai")
        except ImportError:
            print("OpenAI not available - langchain.llms.OpenAI and langchain_openai.OpenAI not found")
except Exception as e:
    print(f"Error importing OpenAI: {e}")
    OPENAI_AVAILABLE = False

# Option 3: Ollama
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

# Option 4: SambaNova
try:
    from langchain_community.chat_models import ChatSambaNovaCloud
    SAMBANOVA_AVAILABLE = True
except ImportError:
    SAMBANOVA_AVAILABLE = False

# Option 5: Google Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatGoogleGenerativeAI
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False


class ChatModelAdapter(LLM):
    """Adapter to make Chat models work like LLM models"""
    chat_model: object = Field(exclude=True)
    
    def __init__(self, chat_model, **kwargs):
        super().__init__(**kwargs)
        self.chat_model = chat_model
    
    @property
    def _llm_type(self) -> str:
        return getattr(self.chat_model, "_llm_type", "chat_model")
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        try:
            from langchain.schema import HumanMessage
        except ImportError:
            from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = self.chat_model.invoke(messages)
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)


class LocalLLM(LLM):
    """Local LLM using transformers"""
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")
    pipeline: Optional[object] = Field(default=None, exclude=True)
    tokenizer: Optional[object] = Field(default=None, exclude=True)
    model: Optional[object] = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7
            )

    @property
    def _llm_type(self) -> str:
        return "local_transformers"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if not self.pipeline:
            return "LLM not initialized"

        result = self.pipeline(prompt, return_full_text=False)
        return result[0]["generated_text"]


def get_llm():
    """Get LLM instance based on configuration"""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    print(f"Initializing LLM with provider: {provider}")

    if provider == "openai":
        if not OPENAI_AVAILABLE or OPENAI_CLASS is None:
            raise RuntimeError("OpenAI not available. Install langchain or langchain-openai package.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        try:
            # Try with openai_api_key parameter first
            try:
                llm = OPENAI_CLASS(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=api_key)
            except TypeError:
                # If that fails, try without the parameter (might use env var)
                llm = OPENAI_CLASS(temperature=0.7, model_name="gpt-3.5-turbo")
            print("OpenAI LLM initialized successfully")
            return llm
        except Exception as e:
            raise RuntimeError(f"Error initializing OpenAI: {e}")

    elif provider == "ollama":
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install langchain-community package.")
        # Default to Docker service name 'ollama' if not specified
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        model_name = os.getenv("LLM_MODEL_NAME", "mistral")
        try:
            print(f"Initializing Ollama with base_url: {base_url}, model: {model_name}")
            llm = Ollama(base_url=base_url, model=model_name, temperature=0)
            print(f"Ollama LLM initialized successfully with model: {model_name} at {base_url}")
            return llm
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            raise RuntimeError(f"Error initializing Ollama with model {model_name} at {base_url}: {e}. Make sure Ollama is running and the model is pulled.")

    elif provider == "sambanova":
        if not SAMBANOVA_AVAILABLE:
            raise RuntimeError("SambaNova not available. Install langchain-community package.")
        model_name = os.getenv("LLM_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct")
        api_key = os.getenv("SAMBANOVA_API_KEY")
        if not api_key:
            raise RuntimeError("SAMBANOVA_API_KEY environment variable is required for SambaNova provider.")
        try:
            # Try different parameter names for API key
            try:
                chat_model = ChatSambaNovaCloud(model=model_name, temperature=0, sambanova_api_key=api_key)
            except TypeError:
                try:
                    chat_model = ChatSambaNovaCloud(model=model_name, temperature=0, api_key=api_key)
                except TypeError:
                    # Some versions might use environment variable
                    os.environ["SAMBANOVA_API_KEY"] = api_key
                    chat_model = ChatSambaNovaCloud(model=model_name, temperature=0)
            llm = ChatModelAdapter(chat_model=chat_model)
            print(f"SambaNova LLM initialized successfully with model: {model_name}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Error initializing SambaNova: {e}")

    elif provider == "gemini":
        if not GEMINI_AVAILABLE:
            raise RuntimeError("Gemini not available. Install langchain-google-genai or langchain-community package.")
        model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash-001")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
        try:
            # Try different parameter names for API key
            try:
                chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)
            except TypeError:
                try:
                    chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=api_key)
                except TypeError:
                    # Some versions might use environment variable
                    os.environ["GOOGLE_API_KEY"] = api_key
                    chat_model = ChatGoogleGenerativeAI(model=model_name, temperature=0)
            llm = ChatModelAdapter(chat_model=chat_model)
            print(f"Gemini LLM initialized successfully with model: {model_name}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Error initializing Gemini: {e}")

    elif provider == "local":
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available. Install transformers and torch packages.")
        model_name = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        try:
            llm = LocalLLM(model_name=model_name)
            print(f"Local LLM initialized successfully with model: {model_name}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Error initializing Local LLM: {e}")

    else:
        raise RuntimeError(
            f"Unknown LLM provider: {provider}. "
            "Supported providers: openai, ollama, sambanova, gemini, local"
        )

