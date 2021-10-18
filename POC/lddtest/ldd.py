#!/usr/bin/env python

import subprocess
import shutil
import sys
import os


def ldd(*filenames):
    libs = set()
    for x in filenames:
        p = subprocess.Popen(['ldd', x], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for x in p.stdout.readlines():
            x = x.decode()
            s = x.split()
            if '=>' in x:
                if len(s) != 3:
                    libs.add(s[2])
    return libs


'''
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('usage: {} filename1 filename2 ...'.format(sys.argv[0]))
    else:
        print('\n'.join(ldd(*sys.argv[1:])))
'''


whitelist = set(os.path.basename(x.strip()) for x in '''
    /usr/lib/libEGL.so
    /usr/lib/libEGL.so.1
    /usr/lib/libEGL.so.1.1.0
    /usr/lib/libGL.so
    /usr/lib/libGL.so.1
    /usr/lib/libGL.so.1.7.0
    /usr/lib/libGLESv2.so
    /usr/lib/libGLESv2.so.2
    /usr/lib/libGLESv2.so.2.1.0
    /usr/lib/libGLX.so
    /usr/lib/libGLX.so.0
    /usr/lib/libGLX.so.0.0.0
    /usr/lib/libGLdispatch.so
    /usr/lib/libGLdispatch.so.0
    /usr/lib/libGLdispatch.so.0.0.0
    /usr/lib/libOpenGL.so
    /usr/lib/libOpenGL.so.0
    /usr/lib/libOpenGL.so.0.0.0
    /usr/lib/libX11.so
    /usr/lib/libX11.so.6
    /usr/lib/libX11.so.6.4.0
    /usr/lib/libX11-xcb.so
    /usr/lib/libX11-xcb.so.1
    /usr/lib/libX11-xcb.so.1.0.0
    /usr/lib/libXau.so
    /usr/lib/libXau.so.6
    /usr/lib/libXau.so.6.0.0
    /usr/lib/libXaw6.so
    /usr/lib/libXaw6.so.6
    /usr/lib/libXaw6.so.6.0.1
    /usr/lib/libXaw7.so
    /usr/lib/libXaw7.so.7
    /usr/lib/libXaw7.so.7.0.0
    /usr/lib/libXaw.so
    /usr/lib/libXaw.so.6
    /usr/lib/libXaw.so.7
    /usr/lib/libXcomposite.so
    /usr/lib/libXcomposite.so.1
    /usr/lib/libXcomposite.so.1.0.0
    /usr/lib/libXcursor.so
    /usr/lib/libXcursor.so.1
    /usr/lib/libXcursor.so.1.0.2
    /usr/lib/libXdamage.so
    /usr/lib/libXdamage.so.1
    /usr/lib/libXdamage.so.1.1.0
    /usr/lib/libXdmcp.so
    /usr/lib/libXdmcp.so.6
    /usr/lib/libXdmcp.so.6.0.0
    /usr/lib/libXext.so
    /usr/lib/libXext.so.6
    /usr/lib/libXext.so.6.4.0
    /usr/lib/libXfixes.so
    /usr/lib/libXfixes.so.3
    /usr/lib/libXfixes.so.3.1.0
    /usr/lib/libXfont2.so
    /usr/lib/libXfont2.so.2
    /usr/lib/libXfont2.so.2.0.0
    /usr/lib/libXft.so
    /usr/lib/libXft.so.2
    /usr/lib/libXft.so.2.3.4
    /usr/lib/libXinerama.so
    /usr/lib/libXinerama.so.1
    /usr/lib/libXinerama.so.1.0.0
    /usr/lib/libXi.so
    /usr/lib/libXi.so.6
    /usr/lib/libXi.so.6.1.0
    /usr/lib/libXmu.so
    /usr/lib/libXmu.so.6
    /usr/lib/libXmu.so.6.2.0
    /usr/lib/libXmuu.so
    /usr/lib/libXmuu.so.1
    /usr/lib/libXmuu.so.1.0.0
    /usr/lib/libXpm.so
    /usr/lib/libXpm.so.4
    /usr/lib/libXpm.so.4.11.0
    /usr/lib/libXpresent.so
    /usr/lib/libXpresent.so.1
    /usr/lib/libXpresent.so.1.0.0
    /usr/lib/libXrandr.so
    /usr/lib/libXrandr.so.2
    /usr/lib/libXrandr.so.2.2.0
    /usr/lib/libXrApiLayer_api_dump.so
    /usr/lib/libXrApiLayer_core_validation.so
    /usr/lib/libXrender.so
    /usr/lib/libXrender.so.1
    /usr/lib/libXrender.so.1.3.0
    /usr/lib/libXRes.so
    /usr/lib/libXRes.so.1
    /usr/lib/libXRes.so.1.0.0
    /usr/lib/libXss.so
    /usr/lib/libXss.so.1
    /usr/lib/libXss.so.1.0.0
    /usr/lib/libXt.so
    /usr/lib/libXt.so.6
    /usr/lib/libXt.so.6.0.0
    /usr/lib/libXtst.so
    /usr/lib/libXtst.so.6
    /usr/lib/libXtst.so.6.1.0
    /usr/lib/libXvMC.so
    /usr/lib/libXvMC.so.1
    /usr/lib/libXvMC.so.1.0.0
    /usr/lib/libXvMCW.so
    /usr/lib/libXvMCW.so.1
    /usr/lib/libXvMCW.so.1.0.0
    /usr/lib/libXv.so
    /usr/lib/libXv.so.1
    /usr/lib/libXv.so.1.0.0
    /usr/lib/libXxf86vm.so
    /usr/lib/libXxf86vm.so.1
    /usr/lib/libXxf86vm.so.1.0.0
    /usr/lib64/ld-linux-x86-64.so.2
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
    '''.strip().splitlines() if x.strip())




def linkdeps(*filenames):
    np = p = set(filenames)
    while np:
        np = ldd(*np)
        np = set(x for x in np if os.path.basename(x) not in whitelist)
        np -= p
        p |= np
    p -= set(filenames)
    return p


executable = 'exe'
for library in linkdeps(executable):
    shutil.copyfile(library, os.path.join(os.path.dirname(executable), os.path.dirname(library)))
