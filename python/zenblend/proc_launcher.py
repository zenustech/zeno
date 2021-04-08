import tempfile
import subprocess
import threading
import sys
import os


preloads = '''import zen
zen.loadLibrary('/home/bate/Develop/zensim/build/FastFLIP/libFLIPlib.so')
'''


def _patch_ld_preload(*pathes):
    for path in pathes:
        path = os.path.realpath(path)
        if os.path.isfile(path):
            break
    else:
        print(f'[ZenEdit] cannot find any one in {pathes}! giving up')
        return

    print(f'[ZenEdit] adding {path} to LD_PRELOAD...')
    ld_preload = os.environ.get('LD_PRELOAD', '')
    if ld_preload:
        ld_preload = path + ':' + ld_preload
    else:
        ld_preload = path
    os.environ['LD_PRELOAD'] = ld_preload


_patch_ld_preload(
        '/usr/lib/libtbbmalloc_proxy.so.2',
        '/usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so.2',
        '/usr/lib/libtbbmalloc_proxy.so',
        '/usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so',
        '/usr/local/lib/libtbbmalloc_proxy.so.2',
        '/usr/local/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so.2',
        '/usr/local/lib/libtbbmalloc_proxy.so',
        '/usr/local/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so',
    )


def get_node_descriptors():
    src = f'''{preloads}
descs = zen.dumpDescriptors()
print('=--=', end='')
print(descs, end='')
print('=--=', end='')
'''
    descs = _run_script(src, capture_output=True).split(b'=--=')[1]
    print('[ZenEdit] found node descriptors:')
    print('=========')
    print(descs.decode(), end='')
    print('=========')
    return descs


def _run_script(src, capture_output=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)

        if capture_output:
            return subprocess.check_output([sys.executable, path])
        else:
            return subprocess.check_call([sys.executable, path])


def _execute_script(src, nframes=1):
    print('[ZenEdit] launching Python script:')
    print('=========')
    src = f'''{preloads}
{src}
for frame in range({nframes}):
\tprint('[Zen] executing frame', frame)
\texecute(frame)
'''
    print(src, end='')
    print('=========')

    _run_script(src)
    print('[ZenEdit] Python process exited')


def execute_script(*args):
    t = threading.Thread(target=_execute_script, args=args, daemon=True)
    t.start()
