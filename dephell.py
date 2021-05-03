import subprocess as sp
import shutil
import os


def ldd(file):
    res = []
    p = sp.Popen(['ldd', file], stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = p.communicate()
    for line in stdout.decode().splitlines():
        try:
            line = line.split(' (')[0]
            line = line.split('\t')[1]
        except ValueError:
            continue
        try:
            name, path = line.split(' => ')
        except ValueError:
            continue
        res.append((name, path))
    return res


os.chdir('/tmp/dso')
had = False
while True:
    for file in os.listdir('.'):
        print('')
        print(file, ':', sep='')
        res = ldd(file)
        for name, path in res:
            assert path != 'not found', (file, name)
            if not os.path.exists(name):
                print('  ', name, '- copying')
                shutil.copy(path, '.')
                had = True
            else:
                print('  ', name, '- exists')
    if not had:
        break
    had = False
