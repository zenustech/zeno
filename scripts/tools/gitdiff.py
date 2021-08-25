#!/usr/bin/env python

import sys
from git import Repo

repo = Repo('.')
ref = sys.argv[1] if len(sys.argv) > 1 else None
files = repo.index.diff(ref)
print('\n'.join(x.a_path for x in files))
