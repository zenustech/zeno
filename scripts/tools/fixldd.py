import subprocess as sp
import shutil
import glob
import os

# pacman -Ql gcc-libs glibc | cut -f2 -d\  | grep '^\/usr\/lib\/lib.*\.so' #| cut -d/ -f4
whitelist = set(x.strip() for x in '''
/usr/lib/libasan.so
/usr/lib/libasan.so.6
/usr/lib/libasan.so.6.0.0
/usr/lib/libatomic.so
/usr/lib/libatomic.so.1
/usr/lib/libatomic.so.1.2.0
/usr/lib/libgcc_s.so
/usr/lib/libgcc_s.so.1
/usr/lib/libgdruntime.so
/usr/lib/libgdruntime.so.2
/usr/lib/libgdruntime.so.2.0.0
/usr/lib/libgfortran.so
/usr/lib/libgfortran.so.5
/usr/lib/libgfortran.so.5.0.0
/usr/lib/libgo.so
/usr/lib/libgo.so.19
/usr/lib/libgo.so.19.0.0
/usr/lib/libgomp.so
/usr/lib/libgomp.so.1
/usr/lib/libgomp.so.1.0.0
/usr/lib/libgphobos.so
/usr/lib/libgphobos.so.2
/usr/lib/libgphobos.so.2.0.0
/usr/lib/libitm.so
/usr/lib/libitm.so.1
/usr/lib/libitm.so.1.0.0
/usr/lib/liblsan.so
/usr/lib/liblsan.so.0
/usr/lib/liblsan.so.0.0.0
/usr/lib/libobjc.so
/usr/lib/libobjc.so.4
/usr/lib/libobjc.so.4.0.0
/usr/lib/libquadmath.so
/usr/lib/libquadmath.so.0
/usr/lib/libquadmath.so.0.0.0
/usr/lib/libstdc++.so
/usr/lib/libstdc++.so.6
/usr/lib/libstdc++.so.6.0.29
/usr/lib/libtsan.so
/usr/lib/libtsan.so.0
/usr/lib/libtsan.so.0.0.0
/usr/lib/libubsan.so
/usr/lib/libubsan.so.1
/usr/lib/libubsan.so.1.0.0
/usr/lib/libBrokenLocale-2.33.so
/usr/lib/libBrokenLocale.so
/usr/lib/libBrokenLocale.so.1
/usr/lib/libSegFault.so
/usr/lib/libanl-2.33.so
/usr/lib/libanl.so
/usr/lib/libanl.so.1
/usr/lib/libc-2.33.so
/usr/lib/libc.so
/usr/lib/libc.so.6
/usr/lib/libcrypt-2.33.so
/usr/lib/libcrypt.so.1
/usr/lib/libdl-2.33.so
/usr/lib/libdl.so
/usr/lib/libdl.so.2
/usr/lib/libm-2.33.so
/usr/lib/libm.so
/usr/lib/libm.so.6
/usr/lib/libmemusage.so
/usr/lib/libmvec-2.33.so
/usr/lib/libmvec.so
/usr/lib/libmvec.so.1
/usr/lib/libnsl-2.33.so
/usr/lib/libnsl.so.1
/usr/lib/libnss_compat-2.33.so
/usr/lib/libnss_compat.so
/usr/lib/libnss_compat.so.2
/usr/lib/libnss_db-2.33.so
/usr/lib/libnss_db.so
/usr/lib/libnss_db.so.2
/usr/lib/libnss_dns-2.33.so
/usr/lib/libnss_dns.so
/usr/lib/libnss_dns.so.2
/usr/lib/libnss_files-2.33.so
/usr/lib/libnss_files.so
/usr/lib/libnss_files.so.2
/usr/lib/libnss_hesiod-2.33.so
/usr/lib/libnss_hesiod.so
/usr/lib/libnss_hesiod.so.2
/usr/lib/libpcprofile.so
/usr/lib/libpthread-2.33.so
/usr/lib/libpthread.so
/usr/lib/libpthread.so.0
/usr/lib/libresolv-2.33.so
/usr/lib/libresolv.so
/usr/lib/libresolv.so.2
/usr/lib/librt-2.33.so
/usr/lib/librt.so
/usr/lib/librt.so.1
/usr/lib/libthread_db-1.0.so
/usr/lib/libthread_db.so
/usr/lib/libthread_db.so.1
/usr/lib/libutil-2.33.so
/usr/lib/libutil.so
/usr/lib/libutil.so.1
'''.strip().splitlines())

resolved = {}
visited = set()

def touch(path):
    if path in whitelist:
        return
    if path in visited:
        return
    visited.add(path)
    print('ldd', path)
    output = sp.check_output(['ldd', path]).decode()
    for line in output.splitlines():
        if '=>' not in line: continue
        lhs, rhs = line.split('=>')
        rhs = rhs.split('(')[0]
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not rhs: continue
        print('{} => {}'.format(lhs, rhs))
        if lhs in resolved:
            assert rhs == resolved[lhs], (rhs, resolved[lhs])
        else:
            resolved[lhs] = rhs
        touch(rhs)

for path in glob.glob('ZenoBin/*.so*'):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        touch(path)

shutil.rmtree('ZenoBin/extra', ignore_errors=True)
os.mkdir('ZenoBin/extra')

for path in visited:
    if 'ZenoBin' not in path:
        name = os.path.basename(path)
        dstpath = os.path.join('ZenoBin/extra', name)
        print('copying', path, '=>', dstpath)
        shutil.copy(path, dstpath)
