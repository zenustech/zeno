import threading


class LazyImport:
    def __init__(self, name):
        self.__name = name
        self.__module = None

    def __getattr__(self, attr):
        if self.__module is None:
            print('* LazyImport:', self.__name)
            self.__module = __import__(self.__name)
        return getattr(self.__module, attr)


def go(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t
