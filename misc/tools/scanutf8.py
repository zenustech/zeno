#!/usr/bin/env python
import sys
import locale
import subprocess

failed = False
def check(path):
    global failed
    try:
        with open(path, 'r') as f:
            for n, bs in enumerate(f.readlines()):
                for b in map(ord, bs):
                    if not 0 <= b <= 0x7f:
                        print('{}:{}: (U+{:04X}) {}'.format(path, n + 1, b, chr(b)))
                        failed = True
    except UnicodeDecodeError:
        print('{}: failed to decode as {}'.format(path, locale.getdefaultlocale()))
        failed = True

def escape(path):
    out = ''
    changed = False
    with open(path, 'rb') as f:
        for bs in f:
            for b in bs:
                if not 0 <= b <= 0x7f:
                    # print('{}:{}: 0x{:02X}'.format(path, n + 1, b))
                    out += '\\x{:02X}'.format(b)
                    changed = True
                else:
                    out += chr(b)
    if changed:
        print(path)
        if not os.environ.get('DRYRUN'):
            with open(path, 'w') as f:
                f.write(out)

process = check
if len(sys.argv) > 1:
    process = {'check': check, 'escape': escape}[sys.argv[1]]

if len(sys.argv) > 2:
    process(sys.argv[2])
else:
    out = subprocess.check_output(['bash', '-c', r"find ui zeno zenovis -type f -regex '.*\.\(c\|h\)\(pp\)?'"])
    for x in out.decode().splitlines():
        process(x)
if failed:
    exit(1)
