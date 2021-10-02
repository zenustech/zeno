import os

path = 'zeno/UI/main.cpp'

with open(path, 'r') as f:
    lines = f.readlines()

    for line in lines:
        line = line.strip('\n')
        if line.startswith('#include "'):
            inc = line.split('"')[1]
            inc = os.path.join(os.path.dirname(path), inc)
            line = '#include <' + inc + '>'
        print(line)
