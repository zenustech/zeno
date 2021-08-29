#O=arts/flip.zsg
O=arts/testprimdup.zsg
#O=arts/testnumvecop.zsg
#O=arts/pa2ls.zsg
#O=arts/testkillpars.zsg
#O=arts/tmptutvdb2.zsg
#O=arts/embeddeform.zsg
#O=arts/prim.zsg
#O=arts/testlitsock.zsg
#O=graphs/BulletRigidSim.zsg
#O=graphs/Xuben_ZFX_IISPH.zsg
#O=arts/ZFXv2.zsg
#O=arts/lowResMPM.zsg
#O=arts/literialconst.zsg
#O=arts/blendtest.zsg
default: justrun
#default: run

#####################################################
###### THIS FILE IS USED BY ARCHIBATE AND ZHXX ######
## NORMAL USERS SHOULD USE `make -C build` INSTEAD ##
#####################################################

#O=arts/ZFXv2.zsg
#default: run

all:
	cmake -B build
	cmake --build build --parallel

release_all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install
	cmake --build build --parallel

debug_all:
	cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/tmp/tmp-install
	cmake --build build --parallel

configure:
	cmake -B build
	ccmake -B build

test: all
	build/tests/zentest

easygl: all
	build/projects/EasyGL/zeno_EasyGL_main

run: all
	ZEN_OPEN=$O python3 -m zenqt

glrun: all
	ZEN_NOFORK=1 ZEN_NOVIEW=1 ZEN_OPEN=$O python3 -m zenqt

justrun:
	ZEN_OPEN=$O python3 -m zenqt

gldebug: debug_all
	ZEN_NOSIGHOOK=1 ZEN_NOVIEW=1 USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O gdb python3 -ex 'r -m zenqt'

debug: debug_all
	ZEN_NOSIGHOOK=1 USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O gdb python3 -ex 'r -m zenqt'

.PHONY: all debug_all debug run test configure default
