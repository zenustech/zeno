import sys, json
from zenqt.main import main
from zenqt.system.main import main as sys_main


if len(sys.argv) > 1:
    # cli only
    #   dump-descs or open .zsg file
    sys.exit(sys_main())
else:
    # GUI
    sys.exit(main())
