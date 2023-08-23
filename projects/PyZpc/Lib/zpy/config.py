import os 

zeno_lib_dir = os.getenv('ZENO_LIB_DIR')
# TODO: handle windows... 
zeno_lib_path = None if zeno_lib_dir is None \
    else os.path.join(zeno_lib_dir, 'libzeno.so')
