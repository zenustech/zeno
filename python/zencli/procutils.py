import os
import sys
import tempfile
import subprocess
import threading


def run_script(src, capture_output=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'script.py')
        with open(path, 'w') as f:
            f.write(src)

        if capture_output:
            return subprocess.check_output([sys.executable, path])
        else:
            return subprocess.check_call([sys.executable, path])


def inject_ld_preload(*pathes):
    for path in pathes:
        path = os.path.realpath(path)
        if os.path.isfile(path):
            break
    else:
        return

    ld_preload = os.environ.get('LD_PRELOAD', '')
    if ld_preload:
        ld_preload = path + ':' + ld_preload
    else:
        ld_preload = path
    os.environ['LD_PRELOAD'] = ld_preload


def go(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
