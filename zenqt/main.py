import sys


def main():
    if len(sys.argv) > 1:
        from .system.main import main as _main
        return _main()
    else:
        from .ui.main import main as _main
        return _main()
