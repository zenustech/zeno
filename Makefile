#A=-DComputeCpp_DIR=/opt/ComputeCpp-CE -DCOMPUTECPP_BITCODE=ptx64

run: all
	$B build/zeno

debug: all
	$B gdb build/zeno -ex r

all:
	test -d /tmp/zeno-ccache || mkdir /tmp/zeno-ccache
	test -d ~/.cache/ccache || ln -sf /tmp/zeno-ccache ~/.cache/ccache
	test -d /tmp/zeno-build || mkdir /tmp/zeno-build
	test -d build || ln -sf /tmp/zeno-build build
	cmake -Wno-dev -B /tmp/zeno-build $A
	cmake --build /tmp/zeno-build --parallel $<

config:
	ccmake -B /tmp/zeno-build

clean:
	rm -rf /tmp/zeno-build
