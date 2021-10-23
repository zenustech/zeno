run: zs_editor
	build/editor/zs_editor

debug: zs_editor
	gdb build/editor/zs_editor -ex r

test: zs_tests
	build/tests/zs_tests

%:
	test -d /tmp/zeno-ccache || mkdir /tmp/zeno-ccache
	test -d ~/.cache/ccache || ln -sf /tmp/zeno-ccache ~/.cache/ccache
	test -d /tmp/zeno-build || mkdir /tmp/zeno-build
	test -d build || ln -sf /tmp/zeno-build build
	cmake -B /tmp/zeno-build
	cmake --build /tmp/zeno-build --parallel

config:
	ccmake -B /tmp/zeno-build
