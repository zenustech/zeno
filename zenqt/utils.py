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
    return chain(capital_search(pattern, keys), direct_search(pattern, keys))

from zenutils import rel2abs

def asset_path(name):
    return rel2abs(__file__, 'assets', name)

