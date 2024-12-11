from llama_cpp import Llama
from configure_llama import configure_llama_model, initialize_llama
from huggingface_hub import hf_hub_download
import torch

class LlamaHandler:
    def __init__(self, model_size: str = "2-7b"):
        """
        Initialisiert den Llama Handler
        
        Args:
            model_size: "3.1-8b", "3.2-3b", "2-7b", "2-13b", "2-70b" für verschiedene Modellgrößen
        """
        # Unterdrücke die FutureWarning Meldung für den Tokenizer
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module='transformers.tokenization_utils_base')
        
        print("Initializing Llama Handler...")
        
        # Definiere Prompt-Templates für verschiedene Modellversionen
        self.prompt_templates = {
            "3.1-8b": """<|system|>{system_prompt}</|system|>
<|user|>{query}</|user|>
<|assistant|>""",
            
            "3.2-3b": """<|system|>{system_prompt}</|system|>
<|user|>{query}</|user|>
<|assistant|>""",
            
            "2-7b": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{query} [/INST]""",
            
            "2-13b": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{query} [/INST]""",
            
            "2-70b": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{query} [/INST]"""
        }
        
        # Modell-IDs für verschiedene Größen
        model_ids = {
            "3.1-8b": "TheBloke/Llama-3.1-8B-GGUF",
            "3.2-3b": "TheBloke/Llama-3.2-3B-GGUF",
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
        
        try:
            # Download des Modells von Hugging Face
            model_path = hf_hub_download(
                repo_id=model_ids[model_size],
                filename=model_files[model_size],
                local_dir="./models"
            )
            
            self.model_version = model_size
            
            # Direkte Initialisierung des Llama-Modells mit optimierten Parametern
            self.llm = Llama(
                model_path=model_path,
                n_ctx=512,          # Reduzierter Kontext
                n_batch=512,        # Erhöhte Batch-Größe
                n_threads=8,        # Mehr Threads für Parallelisierung
                n_gpu_layers=32     # Mehr GPU-Layer für schnellere Verarbeitung
            )
            
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
                max_tokens=256,      # Reduzierte maximale Token-Länge
                temperature=0.7,
                top_p=0.95,
                top_k=40,           # Top-k sampling für schnellere Generierung
                repeat_penalty=1.1,  # Leichte Wiederholungsstrafe
                stop=["</s>", "</|assistant|>"],
                echo=False,
                stream=False        # Deaktiviere Streaming für schnellere Gesamtantwort
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

    def generate_natural_answer(self, question: str, sparql_answer: str) -> str:
        """
        Generiert eine natürlichsprachliche Antwort basierend auf der Frage und dem SPARQL-Ergebnis.
        
        Args:
            question: Die ursprüngliche Frage in natürlicher Sprache
            sparql_answer: Die Antwort aus der SPARQL-Query
            
        Returns:
            str: Eine formatierte natürlichsprachliche Antwort
        """
        try:
            # Wenn keine Ergebnisse gefunden wurden
            if sparql_answer == "Keine Ergebnisse gefunden.":
                return f"Entschuldigung, ich konnte keine Antwort auf die Frage '{question}' finden."
            
            # Generiere Prompt für das Modell
            system_prompt = """Du bist ein hilfreicher Assistent, der Antworten in natürlicher Sprache formuliert. 
            Verwende die gegebenen Rohdaten, um eine präzise, aber natürlich klingende Antwort zu generieren."""
            
            query = f"""Basierend auf der folgenden Frage und den Rohdaten, generiere eine natürlichsprachliche Antwort:

            Frage: {question}
            Rohdaten: {sparql_answer}

            Formuliere eine präzise, aber natürlich klingende Antwort. Verwende die Rohdaten, aber formuliere sie in einen vollständigen Satz um."""
            
            # Wähle das richtige Prompt-Format
            template = self.prompt_templates.get(
                self.model_version, 
                self.prompt_templates["2-7b"]  # Fallback auf Llama 2 Format
            )
            
            full_prompt = template.format(
                system_prompt=system_prompt,
                query=query
            )
            
            # Generiere Antwort mit dem Modell
            response = self.llm(
                full_prompt,
                max_tokens=128,      # Kürzere Antworten für schnellere Generierung
                temperature=0.7,
                top_p=0.95,
                top_k=40,           # Top-k sampling
                repeat_penalty=1.1,
                stop=["</s>", "</|assistant|>"],
                echo=False,
                stream=False
            )
            
            generated_answer = response['choices'][0]['text'].strip()
            
            return generated_answer
            
        except Exception as e:
            print(f"Fehler bei der Generierung der natürlichen Antwort: {e}")
            # Fallback: Gib die ursprüngliche SPARQL-Antwort zurück
            return sparql_answer