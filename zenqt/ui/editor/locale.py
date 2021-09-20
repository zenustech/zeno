from . import *

translation = {}

@eval('lambda x: x()')
def load_translation():
    translation.clear()
    with open(asset_path('zh-cn.txt'), 'r') as f:
        for line in f.readlines():
            try:
                en, zh = line.split(' ', maxsplit=1)
            except ValueError:
                pass
            translation[en.lower()] = zh

def translate(x):
    return translation.get(x.lower(), x)
