#O=a.zsg
#O=b.zsg
#O=c.zsg
#O=d.zsg
#O=i.zsg
O=j.zsg
#O=arts/testspray.zsg
#O=arts/testbulletsim.zsg   # BulletTools/stub.cpp
#O=arts/testvorosplit.zsg  # cgmesh/PrimitiveVoronoi.cpp
#O=arts/flip.zsg           # FLIPtools/stub.cpp

#O=arts/ZFXv2.zsg
#O=arts/segvtrig.zsg
#O=arts/testtbbreduce.zsg
#O=arts/ZFXv2.zsg
#O=arts/vdbperlinnoise.zsg
#O=arts/flip.zsg
#O=arts/testvorobreak.zsg
#O=arts/visualmarchingtetra.zsg
#O=arts/testprimdup.zsg
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
#default: justrun
#default: optrun
default: run

#####################################################
###### THIS FILE IS USED BY ARCHIBATE AND ZHXX ######
## NORMAL USERS SHOULD USE `make -C build` INSTEAD ##
#####################################################

#O=arts/ZFXv2.zsg
#default: run

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=`which python3` # makexinxinVeryHappy
	cmake --build build --parallel 8

run: all
	ZEN_TIMER=/tmp/timer ZEN_OPEN=$O python3 -m zenqt

debug: all
	ZEN_TIMER=/tmp/timer ZEN_NOSIGHOOK=1 USE_GDB=1 ZEN_SPROC=1 ZEN_OPEN=$O gdb python3 -ex 'r -m zenqt'

optrun: all
	ZEN_TIMER=/tmp/timer ZEN_OPEN=$O optirun python3 -m zenqt

noviewrun: all
	ZEN_TIMER=/tmp/timer ZEN_NOFORK=1 ZEN_NOVIEW=1 ZEN_OPEN=$O python3 -m zenqt

justrun:
	ZEN_OPEN=$O python3 -m zenqt
