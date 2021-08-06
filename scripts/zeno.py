import sys

if len(sys.argv) > 1:
    # cli only
    #   dump-descs or open .zsg file
    from zenqt.system.main import main as _main
    sys.exit(_main())
else:
    # GUI
    from zenqt.main import main as _main
    sys.exit(_main())
