import importlib
import inspect
import time
import os
import sys

class HotReloader:
    def __init__(self, module_name, class_name=None, class_args=None, reload_interval=1):
        self.module_name = module_name
        self.class_name = class_name
        self.class_args = class_args or {}
        self.reload_interval = reload_interval
        self.last_modified = 0
        self.module = None
        self.cached_functions = {}
        self.cached_instance = None
        
        # Initial load
        self.reload_if_changed()
    
    def get_file_modified_time(self):
        module = importlib.import_module(self.module_name)
        file_path = inspect.getfile(module)
        return os.path.getmtime(file_path)
    
    def reload_if_changed(self):
        try:
            current_modified = self.get_file_modified_time()
            
            if current_modified > self.last_modified:
                print(f"Reloading {self.module_name}...")
                
                # Reload the module
                if self.module:
                    self.module = importlib.reload(self.module)
                else:
                    self.module = importlib.import_module(self.module_name)
                
                # Update cached functions
                self.cached_functions = {
                    name: func for name, func in inspect.getmembers(self.module, inspect.isfunction)
                }
                
                # If class_name is specified, create new instance
                if self.class_name:
                    class_def = getattr(self.module, self.class_name)
                    self.cached_instance = class_def(**self.class_args)
                
                self.last_modified = current_modified
                return True
        except Exception as e:
            print(f"Error reloading module: {e}")
        return False
    
    def get_function(self, function_name):
        self.reload_if_changed()
        return self.cached_functions.get(function_name)
    
    def get_instance(self):
        """Returns the cached class instance, reloading if necessary"""
        self.reload_if_changed()
        return self.cached_instance