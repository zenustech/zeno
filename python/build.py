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
    shutil.copytree('zen', os.path.join(tmpdir, 'zen'))
    libpath = os.path.join(repodir, '../build/zen/libzenpy.so')
    assert os.path.exists(libpath), 'please build the library first'
    shutil.copy(libpath, os.path.join(tmpdir, 'zen/libzenpy.so'))
    incpath = os.path.join(repodir, '../zen/include')
    shutil.copytree(incpath, os.path.join(tmpdir, 'zen/include'))
    os.chdir(tmpdir)
    subprocess.check_call([sys.executable, 'setup.py', 'bdist_wheel'])
    os.chdir(repodir)
    res = glob.glob(os.path.join(tmpdir, 'dist/*.whl'))
    assert len(res) == 1, res
    assert os.path.exists(res[0]), res[0]
    whlpath = os.path.join(repodir, '../build', os.path.basename(res[0]))
    shutil.copy(res[0], whlpath)
    assert os.path.exists(whlpath), whlpath
    print('done with', whlpath)
