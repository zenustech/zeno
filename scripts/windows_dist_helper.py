#!/usr/bin/env python3

import os



for fname in os.listdir('zenqt\\bin'):
    if fname.endswith('.lib') or fname.endswith('.exp'):
        dstpath = os.path.join('zenqt\\bin', fname)
        print('removing', dstpath)
        os.unlink(dstpath)
