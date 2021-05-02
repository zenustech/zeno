'''
Core DLL singleton
'''


@eval('lambda x: x()')
def get_core():
    def import_core():
        import zenlibs
        return zenlibs.get_library('libzenpy')

    core = None
    def get_core():
        nonlocal core
        if core is None:
            core = import_core()
        return core

    return get_core
