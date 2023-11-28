import os
import ctypes

def check_folder(path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

class CLib:
    def __init__(self, lib_path = None) -> None:
        if lib_path is not None: 
            self.init_lib(lib_path)
    
    def init_lib(self, lib_path):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

    def register(self, restype, func_name, *argtypes):
        func = getattr(self.lib, func_name)
        func.restype = restype
        func.argtypes = argtypes
    
    def call(self, func_name, *args):
        func = getattr(self.lib, func_name)
        return func(*args)
