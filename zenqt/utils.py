from itertools import chain

def capital_search(pattern, keys):
    pattern = pattern.lower()
    matched = filter(lambda k: capital_match(pattern, k), keys)
    return matched

def capital_match(pattern, key):
    only_upper = ''.join(filter(lambda c: c.isupper(), key)).lower()
    return pattern in only_upper

def direct_search(pattern, keys):
    pattern = pattern.lower()
    matched = filter(lambda k: pattern in k.lower(), keys)
    return matched

def fuzzy_search(pattern, keys):
    front = list(capital_search(pattern, keys))
    front_set = set(front)
    back = direct_search(pattern, keys)
    back = [k for k in back if (k not in front_set)]
    return front + back

from .system.utils import rel2abs

def asset_path(name):
    return rel2abs(__file__, 'assets', name)

def setKeepAspect(renderer):
    if hasattr(renderer, 'setAspectRatioMode'):
        # PySide2 >= 5.15
        from PySide2.QtCore import Qt
        renderer.setAspectRatioMode(Qt.KeepAspectRatio)
    else:
        print('WARNING: setAspectRatioMode failed to work')
