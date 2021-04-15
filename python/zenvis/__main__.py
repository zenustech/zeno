import sys
import os


@eval('lambda x: x()')
def core():
    import os
    import sys

    lib_dir = os.path.dirname(__file__)

    assert os.path.exists(lib_dir)
    assert os.path.exists(os.path.join(lib_dir, 'libzenvis.so'))

    sys.path.insert(0, lib_dir)
    try:
        import libzenvis as core
    finally:
        assert sys.path.pop(0) == lib_dir

    return core


core.initialize()
while core.new_frame():
    pass
core.finalize()
