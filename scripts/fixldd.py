import subprocess as sp

# pacman -Ql gcc-libs glibc | cut -f2 -d\  | grep '^\/usr\/lib\/lib.*\.so' | cut -d/ -f4
whitelist = '''
libasan.so
libasan.so.6
libasan.so.6.0.0
libatomic.so
libatomic.so.1
libatomic.so.1.2.0
libgcc_s.so
libgcc_s.so.1
libgdruntime.so
libgdruntime.so.2
libgdruntime.so.2.0.0
libgfortran.so
libgfortran.so.5
libgfortran.so.5.0.0
libgo.so
libgo.so.19
libgo.so.19.0.0
libgomp.so
libgomp.so.1
libgomp.so.1.0.0
libgphobos.so
libgphobos.so.2
libgphobos.so.2.0.0
libitm.so
libitm.so.1
libitm.so.1.0.0
liblsan.so
liblsan.so.0
liblsan.so.0.0.0
libobjc.so
libobjc.so.4
libobjc.so.4.0.0
libquadmath.so
libquadmath.so.0
libquadmath.so.0.0.0
libstdc++.so
libstdc++.so.6
libstdc++.so.6.0.29
libtsan.so
libtsan.so.0
libtsan.so.0.0.0
libubsan.so
libubsan.so.1
libubsan.so.1.0.0
libBrokenLocale-2.33.so
libBrokenLocale.so
libBrokenLocale.so.1
libSegFault.so
libanl-2.33.so
libanl.so
libanl.so.1
libc-2.33.so
libc.so
libc.so.6
libcrypt-2.33.so
libcrypt.so.1
libdl-2.33.so
libdl.so
libdl.so.2
libm-2.33.so
libm.so
libm.so.6
libmemusage.so
libmvec-2.33.so
libmvec.so
libmvec.so.1
libnsl-2.33.so
libnsl.so.1
libnss_compat-2.33.so
libnss_compat.so
libnss_compat.so.2
libnss_db-2.33.so
libnss_db.so
libnss_db.so.2
libnss_dns-2.33.so
libnss_dns.so
libnss_dns.so.2
libnss_files-2.33.so
libnss_files.so
libnss_files.so.2
libnss_hesiod-2.33.so
libnss_hesiod.so
libnss_hesiod.so.2
libpcprofile.so
libpthread-2.33.so
libpthread.so
libpthread.so.0
libresolv-2.33.so
libresolv.so
libresolv.so.2
librt-2.33.so
librt.so
librt.so.1
libthread_db-1.0.so
libthread_db.so
libthread_db.so.1
libutil-2.33.so
libutil.so
libutil.so.1
'''.strip().splitlines()

file = '/lib/libopenvdb.so.8.1.0'
output = sp.check_output(['ldd', file]).decode()
for line in output.splitlines():
    if '=>' not in line: continue
    lhs, rhs = line.split('=>')
    rhs = rhs.split('(')[0]
    lhs = lhs.strip()
    rhs = rhs.strip()
    print('!{}!{}!'.format(lhs, rhs))
