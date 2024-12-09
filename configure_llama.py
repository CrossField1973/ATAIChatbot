from llama_cpp import Llama
import psutil
import os
from check_hardware import check_system_resources

def configure_llama_model(model_path: str):
    resources = check_system_resources()
    
    # Basis-Konfiguration
    config = {
        'model_path': model_path,
        'n_ctx': 2048,  # Standardkontext
        'n_batch': 512,  # Standard Batch-Größe
    }
    
    # CPU-Konfiguration
    if resources['cpu_threads'] > 4:
        config['n_threads'] = resources['cpu_threads'] - 2  # Reserviere 2 Threads für System
    else:
        config['n_threads'] = max(1, resources['cpu_threads'] - 1)
    
    # RAM-basierte Anpassungen
    if resources['ram_available'] > 16:  # Mehr als 16GB verfügbar
        config['n_ctx'] = 4096  # Größerer Kontext
        config['n_batch'] = 1024
    elif resources['ram_available'] < 8:  # Weniger als 8GB verfügbar
        config['n_ctx'] = 1024  # Kleinerer Kontext
        config['n_batch'] = 256
    
    # GPU-Konfiguration
    if resources['gpu_available'] and resources['gpu_memory'][0] > 4:  # >4GB VRAM
        config['n_gpu_layers'] = -1  # Alle Layer auf GPU
    elif resources['gpu_available'] and resources['gpu_memory'][0] > 2:  # >2GB VRAM
        config['n_gpu_layers'] = 32  # Teilweise GPU-Nutzung
    else:
        config['n_gpu_layers'] = 0  # CPU-only
    
    return config

def initialize_llama(config: dict):
    try:
        model = Llama(**config)
        print(f"""
Llama Model initialized with:
----------------------------
Context Length: {config['n_ctx']}
Batch Size: {config['n_batch']}
CPU Threads: {config['n_threads']}
GPU Layers: {config['n_gpu_layers']}
        """)
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

if __name__ == "__main__":
    model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    config = configure_llama_model(model_path)
    model = initialize_llama(config) 