import psutil
import torch
import os

def check_system_resources():
    # CPU Info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    
    # RAM Info
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024 ** 3)  # GB
    ram_available = ram.available / (1024 ** 3)  # GB
    
    # GPU Info
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
    gpu_memory = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(gpu_count)] if gpu_available else []
    
    print(f"""
System Resources:
----------------
CPU Cores (Physical): {cpu_count}
CPU Threads: {cpu_threads}
RAM Total: {ram_total:.2f} GB
RAM Available: {ram_available:.2f} GB
GPU Available: {gpu_available}
GPU Count: {gpu_count}
GPU Names: {gpu_names}
GPU Memory: {[f'{mem:.2f} GB' for mem in gpu_memory]}
    """)
    
    return {
        'cpu_count': cpu_count,
        'cpu_threads': cpu_threads,
        'ram_total': ram_total,
        'ram_available': ram_available,
        'gpu_available': gpu_available,
        'gpu_count': gpu_count,
        'gpu_memory': gpu_memory
    }

if __name__ == "__main__":
    resources = check_system_resources() 