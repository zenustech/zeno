#A=-DComputeCpp_DIR=/opt/ComputeCpp-CE -DCOMPUTECPP_BITCODE=ptx64
A=-DCMAKE_CXX_COMPILER=/opt/sycl/bin/clang++ -DSYCL_TARGETS=nvptx64-nvidia-cuda

run: zeno_editor
	build/editor/zeno_editor

debug: zeno_editor
	gdb build/editor/zeno_editor -ex r

test: zeno_tests
	build/tests/zeno_tests

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
