from itertools import chain


def key_appear_by_order(pattern, key):
    key = key.lower()
    for c in pattern:
        res = key.find(c)
        if res == -1:
            return False
        key = key[res + 1:]
    return True

def capital_match(pattern, key):
    only_upper = ''.join(filter(lambda c: c.isupper(), key)).lower()
    return pattern in only_upper

# remove duplicate and keep order
def merge_condidates(*lsts):
    res_list = []
    res_set = set()
    for lst in lsts:
        res_list.extend([k for k in lst if k not in res_set])
        res_set = set(res_list)
    return res_list

def fuzzy_search(pattern, keys):
    pattern = pattern.lower()
    key_appear_by_order_conds = [k for k in keys if key_appear_by_order(pattern, k)]
    direct_match_conds = [k for k in key_appear_by_order_conds if pattern in k.lower()]
    prefix_match_conds = [k for k in direct_match_conds if k.lower().startswith(pattern)]
    capital_match_conds = [k for k in key_appear_by_order_conds if capital_match(pattern, k)]

    return merge_condidates(
        prefix_match_conds,
        capital_match_conds,
        direct_match_conds,
        key_appear_by_order_conds
    )

def setKeepAspect(renderer):
    if hasattr(renderer, 'setAspectRatioMode'):
        # PySide2 >= 5.15
        from PySide2.QtCore import Qt
        renderer.setAspectRatioMode(Qt.KeepAspectRatio)
    else:
        print('WARNING: setAspectRatioMode failed to work')


def asset_path(name):
    from ..utils import relative_path
    return relative_path('assets', name)
