libdir = os.path.dirname(os.path.abspath(__file__))


def get_library(name):
    from zenutils import import_library
    return import_library(libdir, name)
