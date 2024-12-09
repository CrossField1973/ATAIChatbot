from llama_cpp import Llama
from configure_llama import configure_llama_model, initialize_llama
from huggingface_hub import hf_hub_download

class LlamaHandler:
    def __init__(self, model_size: str = "2-7b"):
        """
        Initialisiert den Llama Handler
        
        Args:
            model_size: "3.1-8b", "3.2-3b", "2-7b", "2-13b", "2-70b" für verschiedene Modellgrößen
        """
        print("Initializing Llama Handler...")
        
        # Modell-IDs für verschiedene Größen
        model_ids = {
            "3.1-8b": "TheBloke/Llama-3.1-8B-GGUF",  # Neue Llama 3.1 Version
            "3.2-3b": "TheBloke/Llama-3.2-3B-GGUF",  # Neue Llama 3.2 Version
            "2-7b": "TheBloke/Llama-2-7B-Chat-GGUF",
            "2-13b": "TheBloke/Llama-2-13B-Chat-GGUF",
            "2-70b": "TheBloke/Llama-2-70B-Chat-GGUF"
        }
        
        model_files = {
            "3.1-8b": "llama-3.1-8b.Q4_K_M.gguf",
            "3.2-3b": "llama-3.2-3b.Q4_K_M.gguf",
            "2-7b": "llama-2-7b-chat.Q4_K_M.gguf",
            "2-13b": "llama-2-13b-chat.Q4_K_M.gguf",
            "2-70b": "llama-2-70b-chat.Q4_K_M.gguf"
        }
        
        # Prompt-Formate für verschiedene Modellversionen
        self.prompt_templates = {
            "3.1-8b": """<|system|>
            {system_prompt}
            </|system|>
            <|user|>
            {query}
            </|user|>
            <|assistant|>""",
            
            "3.2-3b": """<|system|>
            {system_prompt}
            </|system|>
            <|user|>
            {query}
            </|user|>
            <|assistant|>""",
            
            "2-7b": """<s>[INST] <<SYS>>
            {system_prompt}
            <</SYS>>
            
            {query}[/INST]"""
        }
        
        # Prüfe ob Modellgröße verfügbar
        if model_size not in model_ids:
            raise ValueError(f"Invalid model size. Choose from: {', '.join(model_ids.keys())}")
            
        try:
            # Download des Modells von Hugging Face
            model_path = hf_hub_download(
                repo_id=model_ids[model_size],
                filename=model_files[model_size],
                local_dir="./models",
                local_dir_use_symlinks=False
            )
            
            # Speichere die Modellversion für spätere Verwendung
            self.model_version = model_size
            
            # Konfiguration und Initialisierung
            model_config = configure_llama_model(model_path)
            self.llm = initialize_llama(model_config)
            
            if not self.llm:
                raise RuntimeError("Failed to initialize Llama model")
                
            print(f"Successfully initialized Llama {model_size} model")
            
        except Exception as e:
            print(f"Error initializing Llama model: {e}")
            raise

    def get_response(self, query: str) -> str:
        """
        Generiert eine Antwort mit dem Llama Modell
        """
        try:
            # Angepasster Prompt
            system_prompt = """You are a helpful AI assistant with knowledge about movies. 
            Answer questions about movies, actors, and related topics. 
            If you're not sure about something, say so."""
            
            # Wähle das richtige Prompt-Format basierend auf der Modellversion
            template = self.prompt_templates.get(
                self.model_version, 
                self.prompt_templates["2-7b"]  # Fallback auf Llama 2 Format
            )
            
            full_prompt = template.format(
                system_prompt=system_prompt,
                query=query
            )
            
            response = self.llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.95,
                stop=["</s>", "</|assistant|>"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"Error in LLM response generation: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

    def __call__(self, query: str) -> str:
        """
        Ermöglicht die direkte Verwendung der Klasse als Funktion
        """
        return self.get_response(query)