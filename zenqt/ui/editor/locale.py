from . import *

translation = {}

def load_translation():
    import os
    translation.clear()
    with open(asset_path('language.txt')) as f:
        lang = f.read()
    if lang == 'en':
        return
    with open(asset_path('zh-cn.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                en, zh = line.split(' ', maxsplit=1)
                translation[en.lower()] = zh
            except ValueError:
                pass

load_translation()

def translate(x):
    return translation.get(x.lower(), x)
