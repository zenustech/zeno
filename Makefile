#A=-DComputeCpp_DIR=/opt/ComputeCpp-CE -DCOMPUTECPP_BITCODE=ptx64
A=-DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ -DSYCL_TARGETS=spir64_x86_64#nvptx64-nvidia-cuda
B=LD_LIBRARY_PATH=/opt/sycl/lib
#T=zeno_editor
T=zeno_cliface

run: $T
	$B build/*/$T

debug: $T
	$B gdb build/*/$T -ex r

test: zeno_tests
	$B build/tests/zeno_tests

%:
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
