import tempfile
import subprocess
import threading
import sys
import os


preloads = '''import zen
zen.loadLibrary('build/FastFLIP/libFLIPlib.so')
'''

def add_ld_preload(*pathes):
    for path in pathes:
        path = os.path.realpath(path)
        if os.path.isfile(path):
            break
    else:
        raise RuntimeError(f'Cannot find any one of {pathes}')
    print(f'[ZenEdit] adding {path} to LD_PRELOAD...')
    ld_preload = os.environ.get('LD_PRELOAD', '')
    if ld_preload:
        ld_preload = path + ':' + ld_preload
    else:
        ld_preload = path
    os.environ['LD_PRELOAD'] = ld_preload

add_ld_preload(
        '/usr/lib/libtbbmalloc_proxy.so',
        '/usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so',
    )


@eval('lambda x: x()')
def editor():
    import os
    import sys

    lib_dir = os.path.dirname(__file__)

    assert os.path.exists(lib_dir)
    assert os.path.exists(os.path.join(lib_dir, 'libzeneditor.so'))

    sys.path.insert(0, lib_dir)
    try:
        import libzeneditor as editor
    finally:
        assert sys.path.pop(0) == lib_dir

    return editor


def update_node_descs():
    src = f'''{preloads}
descs = zen.dumpDescriptors()
print('=--=', end='')
print(descs, end='')
print('=--=', end='')
'''
    descs = run_script(src, capture_output=True).split(b'=--=')[1]
    print('[ZenEdit] found node descriptors:')
    print('=========')
    print(descs.decode(), end='')
    print('=========')
    editor.load_descs(descs)

def run_script(src, capture_output=False):
    if 1:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'script.py')
            with open(path, 'w') as f:
                f.write(src)

            if capture_output:
                return subprocess.check_output([sys.executable, path])
            else:
                return subprocess.check_call([sys.executable, path])

    else:
        exec(compile(src, '<script>', 'exec'))

def do_execute_script(src, nframes):
    print('[ZenEdit] launching Python script:')
    print('=========')
    src = f'''{preloads}
{src}
zen.initialize()
for frame in range({nframes}):
\tprint('[Zen] executing frame', frame)
\texecute(frame)
zen.finalize()
'''
    print(src, end='')
    print('=========')

    run_script(src)
    print('[ZenEdit] Python process exited')


def execute_script(*args):
    if 1:
        t = threading.Thread(target=do_execute_script, args=args, daemon=True)
        t.start()
    else:
        return do_execute_script(*args)


print('[ZenEdit] starting editor...')
editor.initialize()
update_node_descs()
while True:
    ret = editor.new_frame()
    if 'close' in ret:
        break
    if 'refresh' in ret:
        update_node_descs()
    if 'execute' in ret:
        execute_script(ret['execute'], ret['exec_nframes'])
editor.finalize()
