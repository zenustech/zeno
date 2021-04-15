@eval('lambda x: x()')
def core():
    import os
    import sys

    lib_dir = os.path.dirname(__file__)
    lib_path = os.path.join(lib_dir, 'libzenvis.so')
    assert os.path.exists(lib_path), lib_path

    sys.path.insert(0, lib_dir)
    try:
        import libzenvis as core
    finally:
        assert sys.path.pop(0) == lib_dir

    return core
