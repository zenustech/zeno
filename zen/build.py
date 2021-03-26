import os
import sys
import glob
import shutil
import tempfile
import subprocess


repodir = os.path.dirname(os.path.abspath(__file__))
os.chdir(repodir)
with tempfile.TemporaryDirectory() as tmpdir:
    shutil.copy('setup.py', tmpdir)
    shutil.copy('MANIFEST.in', tmpdir)
    shutil.copytree('zen', os.path.join(tmpdir, 'zen'))
    assert os.path.exists('build/libzenpy.so'), 'please build the library first'
    shutil.copy('build/libzenpy.so', os.path.join(tmpdir, 'zen/libzenpy.so'))
    shutil.copytree('include', os.path.join(tmpdir, 'zen/include'))
    os.chdir(tmpdir)
    subprocess.check_call([sys.executable, 'setup.py', 'bdist_wheel'])
    os.chdir(repodir)
    res = glob.glob(os.path.join(tmpdir, 'dist', '*.whl'))
    assert len(res) == 1, res
    whlpath = os.path.join(repodir, 'build', os.path.basename(res[0]))
    shutil.copy(res[0], whlpath)
    assert os.path.exists(whlpath)
    print('done with', whlpath)
