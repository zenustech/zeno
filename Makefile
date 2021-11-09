##############################################################
### WARNING: THIS FILE IS ONLY USED BY ARCHIBATE AND ZHXX! ###
### NORMAL USERS SHOULD USE `cmake --build build` INSTEAD! ###
##############################################################

#A=-DComputeCpp_DIR=/opt/ComputeCpp-CE -DCOMPUTECPP_BITCODE=ptx64
#A=-DBATE_SYCL:BOOL=ON

x: run

run: all
	$B build/zeno

debug: all
	$B gdb build/zeno -ex r

all: adhoc
	test -d /tmp/zeno-ccache || mkdir /tmp/zeno-ccache
	test -d ~/.cache/ccache || ln -sf /tmp/zeno-ccache ~/.cache/ccache
	test -d /tmp/zeno-build || mkdir /tmp/zeno-build
	test -d build || ln -sf /tmp/zeno-build build
	cmake -Wno-dev -B /tmp/zeno-build $A
	cmake --build /tmp/zeno-build --parallel 12

config: adhoc
	ccmake -B /tmp/zeno-build

clean: adhoc
	rm -rf /tmp/zeno-build

adhoc:
	@[[ -d /home/bate ]] || [[ -d /home/dilei ]] || (echo "ERROR: Please use 'make -C build' instead of 'make'" && false)
